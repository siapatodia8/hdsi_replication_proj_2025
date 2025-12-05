"""
LangGraph workflow agent for biomedical knowledge graphs.

Workflow (fan-out/fan-in):
Classify → Extract → Generate → (Explain || Execute in parallel) → Format
"""

import json
import os
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .graph_interface import GraphInterface


class WorkflowState(TypedDict, total=False):
    """State that flows through the workflow steps."""
    user_question: str
    question_type: Optional[str]
    entities: Optional[List[str]]
    cypher_query: Optional[str]
    results: Optional[List[Dict]]
    final_answer: Optional[str]
    error: Optional[str]
    query_explanation: Optional[str]  # human-readable explanation of the Cypher query


class WorkflowAgent:
    """LangGraph workflow agent for biomedical knowledge graphs."""

    # Class constants
    MODEL_NAME = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 200

    # Default schema query
    SCHEMA_QUERY = (
        "MATCH (n) RETURN labels(n) as node_type, count(n) as count "
        "ORDER BY count DESC LIMIT 10"
    )

    def __init__(self, graph_interface: GraphInterface, anthropic_api_key: str):
        self.graph_db = graph_interface
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.schema = self.graph_db.get_schema_info()
        self.property_values = self._get_key_property_values()
        self.workflow = self._create_workflow()

    def _get_key_property_values(self) -> Dict[str, List[Any]]:
        """Discover property names and sample values from the graph."""
        values: Dict[str, List[Any]] = {}
        try:
            # Node properties
            for node_label in self.schema.get("node_labels", []):
                node_props = self.schema.get("node_properties", {}).get(node_label, [])
                for prop_name in node_props:
                    if prop_name in values:
                        continue
                    try:
                        prop_values = self.graph_db.get_property_values(node_label, prop_name)
                        if prop_values:
                            values[prop_name] = prop_values
                    except Exception:
                        continue

            # Relationship properties
            for rel_type in self.schema.get("relationship_types", []):
                rel_label = f"REL_{rel_type}"
                rel_props = self.schema.get("relationship_properties", {}).get(rel_type, [])
                for prop_name in rel_props:
                    if prop_name in values:
                        continue
                    try:
                        prop_values = self.graph_db.get_property_values(rel_label, prop_name)
                        if prop_values:
                            values[prop_name] = prop_values
                    except Exception:
                        continue
        except Exception:
            pass
        return values

    def _get_llm_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Call Anthropic and return plain text."""
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        try:
            resp = self.anthropic.messages.create(
                model=self.MODEL_NAME,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.content[0]
            return content.text.strip() if hasattr(content, "text") else str(content)
        except Exception as e:
            raise RuntimeError(f"LLM response failed: {e}")

    def _create_workflow(self) -> Any:
        """Create the LangGraph workflow with parallel branches."""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("classify", self.classify_question)
        workflow.add_node("extract", self.extract_entities)
        workflow.add_node("generate", self.generate_query)

        # Parallel branches
        workflow.add_node("explain", self.explain_query)
        workflow.add_node("execute", self.execute_query)

        # Fan-in
        workflow.add_node("format", self.format_answer)

        workflow.add_edge("classify", "extract")
        workflow.add_edge("extract", "generate")

        # Fan-out from generate
        workflow.add_edge("generate", "explain")
        workflow.add_edge("generate", "execute")

        # Both branches feed into format (soft fan-in inside the node)
        workflow.add_edge("explain", "format")
        workflow.add_edge("execute", "format")

        workflow.add_edge("format", END)

        workflow.set_entry_point("classify")
        return workflow.compile()

    def _build_classification_prompt(self, question: str) -> str:
        return f"""Classify this biomedical question. Choose one:
- gene_disease: genes and diseases
- drug_treatment: drugs and treatments
- protein_function: proteins and functions
- general_db: database exploration
- general_knowledge: biomedical concepts

Question: {question}

Respond with just the type."""

    def classify_question(self, state: WorkflowState) -> WorkflowState:
        """Return only question_type (and error on failure)."""
        try:
            qtype = self._get_llm_response(
                self._build_classification_prompt(state["user_question"]),
                max_tokens=20,
            )
            return {"question_type": qtype}
        except Exception as e:
            return {"question_type": "general_knowledge", "error": f"Classification failed: {e}"}

    def extract_entities(self, state: WorkflowState) -> WorkflowState:
        """Return only entities."""
        qtype = state.get("question_type")
        if qtype in ["general_db", "general_knowledge"]:
            return {"entities": []}

        property_info = []
        for prop_name, values in self.property_values.items():
            if values:
                sample = ", ".join(str(v) for v in values[:3])
                property_info.append(f"- {prop_name}: {sample}")

        entity_types = ", ".join(self.schema.get("node_labels", []))
        rel_types = ", ".join(self.schema.get("relationship_types", []))

        prompt = (
            "Extract biomedical terms and concepts from this question based on the database schema:\n\n"
            f"Available entity types: {entity_types}\n"
            f"Available relationships: {rel_types}\n\n"
            "Available property values in database:\n"
            f"{chr(10).join(property_info) if property_info else '- No property values available'}\n\n"
            f"Question: {state['user_question']}\n\n"
            'Return a JSON list: ["term1", "term2"] or []'
        )

        try:
            text = self._get_llm_response(prompt, max_tokens=100).strip()
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            ents = json.loads(text)
            return {"entities": ents}
        except Exception:
            return {"entities": []}

    def generate_query(self, state: WorkflowState) -> WorkflowState:
        """Return only cypher_query."""
        qtype = state.get("question_type", "general")

        if qtype == "general_db":
            return {"cypher_query": self.SCHEMA_QUERY}

        if qtype == "general_knowledge":
            return {"cypher_query": None}

        relationship_guide = "Available relationships:\n" + "\n".join(
            f"- {rel}" for rel in self.schema["relationship_types"]
        )

        property_details = []
        for prop_name, values in self.property_values.items():
            if values:
                vtype = "text values" if isinstance(values[0], str) else "numeric values"
                property_details.append(f"- {prop_name}: {values} ({vtype})")

        property_info = (
            "Property names and values:\n"
            f"Node properties: {self.schema['node_properties']}\n"
            "Available property values:\n"
            f"{chr(10).join(property_details) if property_details else '- No values available'}\n"
            "Use WHERE property IN [value1, value2] for filtering."
        )

        prompt = (
            "Create a Cypher query for this biomedical question:\n\n"
            f"Question: {state['user_question']}\n"
            f"Type: {qtype}\n"
            "Schema:\n"
            f"Nodes: {', '.join(self.schema['node_labels'])}\n"
            f"Relations: {', '.join(self.schema['relationship_types'])}\n"
            f"{property_info}\n{relationship_guide}\n"
            f"Entities: {state.get('entities', [])}\n\n"
            "Use MATCH, WHERE with CONTAINS for filtering, RETURN, LIMIT 10.\n"
            "IMPORTANT: Use property names from schema above and IN filtering for property values.\n"
            "Return only the Cypher query."
        )

        cypher = self._get_llm_response(prompt, max_tokens=150)
        if cypher.startswith("```"):
            cypher = "\n".join(
                ln for ln in cypher.split("\n")
                if not ln.startswith("```") and not ln.startswith("cypher")
            ).strip()

        return {"cypher_query": cypher}

    def explain_query(self, state: WorkflowState) -> WorkflowState:
        """Return only query_explanation (or {} if nothing to explain)."""
        cq = state.get("cypher_query")
        if not cq:
            return {}
        try:
            prompt = (
                "Explain in simple terms what this Cypher query does and what it will return. "
                "Keep it to 3–5 sentences and avoid jargon.\n\n"
                f"{cq}"
            )
            explanation = self._get_llm_response(prompt, max_tokens=180)
            return {"query_explanation": explanation}
        except Exception as e:
            return {"query_explanation": f"(Could not generate explanation: {e})"}

    def execute_query(self, state: WorkflowState) -> WorkflowState:
        """Return only results (and error on failure)."""
        try:
            cq = state.get("cypher_query")
            results = self.graph_db.execute_query(cq) if cq else []
            return {"results": results}
        except Exception as e:
            return {"results": [], "error": f"Query failed: {e}"}

    def format_answer(self, state: WorkflowState) -> WorkflowState:
        """Soft fan-in: emit final_answer only when ready."""
        if state.get("error"):
            return {"final_answer": f"Sorry, I had trouble with that question: {state['error']}"}

        qtype = state.get("question_type")
        if qtype == "general_knowledge":
            answer = self._get_llm_response(
                f"""Answer this general biomedical question using your knowledge:

Question: {state['user_question']}

Provide a clear, informative answer about biomedical concepts.""",
                max_tokens=300,
            )
            return {"final_answer": answer}

        results = state.get("results", None)          
        # explanation = state.get("query_explanation")  

        if results is None:
            # Not ready; produce no updates (prevents concurrent key conflicts)
            return {}

        if not results:
            msg = ("I didn't find any information for that question. "
                   "Try asking about genes, diseases, or drugs in our database.")
            # if explanation:
            #     msg = f"**What the query does**: {explanation}\n\n{msg}"
            return {"final_answer": msg}

        answer_text = self._get_llm_response(
            f"""Convert these database results into a clear answer:

Question: {state['user_question']}
Results: {json.dumps(results[:5], indent=2)}
Total found: {len(results)}

Make it concise and informative.""",
            max_tokens=250,
        )

        # if explanation:
        #     answer_text = f"**What the query does**: {explanation}\n\n{answer_text}"

        return {"final_answer": answer_text}

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Run the workflow end-to-end."""
        initial_state: WorkflowState = {
            "user_question": question
        }
        final_state = self.workflow.invoke(initial_state)

        return {
            "timestamp": datetime.now().isoformat(),
            "answer": final_state.get("final_answer", "No answer generated"),
            "question_type": final_state.get("question_type"),
            "entities": final_state.get("entities", []),
            "cypher_query": final_state.get("cypher_query"),
            "query_explanation": final_state.get("query_explanation"),
            "results_count": len(final_state.get("results", [])) if final_state.get("results") is not None else 0,
            "raw_results": (final_state.get("results", []) or [])[:3],
            "error": final_state.get("error"),
        }


def create_workflow_graph() -> Any:
    """Factory function for LangGraph Studio."""
    load_dotenv()
    graph_interface = GraphInterface(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )
    agent = WorkflowAgent(
        graph_interface=graph_interface,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )
    return agent.workflow
