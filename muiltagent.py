
import json
import pandas as pd
import numpy as np
from smolagents import ToolCallingAgent, CodeAgent, InferenceClientModel, ManagedAgent

class ResponseEvaluationManager:
    """
    Manager Agent that coordinates 4 specialized evaluation sub-agents:
    1. Clarity and Conciseness Evaluator
    2. Pedagogy Evaluator (LLM-based)
    3. Statistical Response Evaluator (Evidently-based)
    4. Fact Checker
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = {}
        self.sub_agents = {}
        self._create_separate_models()
        self._setup_managed_agents()
        self._create_manager_agent()
        
    def _create_separate_models(self):
        """Create dedicated model instances for each component"""
        
        # Separate model for manager (optimized for coordination and code generation)
        self.models['manager'] = InferenceClientModel(
            # Optional: Configure for code generation tasks
            # model_id="gpt-4-code", temperature=0.1
        )
        
        # Separate model for clarity evaluation
        self.models['clarity'] = InferenceClientModel(
            # Optional: Configure for structured evaluation tasks  
            # model_id="gpt-4", temperature=0.2
        )
        
        # Separate model for pedagogy evaluation
        self.models['pedagogy'] = InferenceClientModel(
            # Optional: Configure for educational assessment
            # model_id="gpt-4", temperature=0.3
        )
        
        # Separate model for statistical evaluation
        self.models['statistics'] = InferenceClientModel(
            # Optional: Configure for analytical tasks
            # model_id="gpt-4", temperature=0.1
        )
        
        # Separate model for fact checking
        self.models['fact_checker'] = InferenceClientModel(
            # Optional: Configure for factual verification
            # model_id="gpt-4", temperature=0.0
        )
        
        print(f"✅ Created {len(self.models)} separate model instances")
        
    def _setup_managed_agents(self):
        """Setup sub-agents with dedicated model instances"""
        
        # 1. Clarity and Conciseness Agent with dedicated model
        clarity_tool = clarityAndConcesseness()
        clarity_agent = ToolCallingAgent(
            tools=[clarity_tool],
            model=self.models['clarity'],  # 🔄 Dedicated model
            name="clarity_agent"
        )
        self.sub_agents['clarity'] = ManagedAgent(
            agent=clarity_agent,
            name="clarity_evaluator", 
            description="Evaluates response clarity and conciseness using structured rubrics."
        )
        
        # 2. Pedagogy Evaluator Agent with dedicated model
        pedagogy_tool = LLMResponsePedagogyEvaluator()
        pedagogy_agent = ToolCallingAgent(
            tools=[pedagogy_tool],
            model=self.models['pedagogy'],  # 🔄 Dedicated model
            name="pedagogy_agent"
        )
        self.sub_agents['pedagogy'] = ManagedAgent(
            agent=pedagogy_agent,
            name="pedagogy_evaluator",
            description="Evaluates pedagogical quality using LLM-based rubrics."
        )
        
        # 3. Statistical Response Evaluator Agent with dedicated model
        stats_tool = EvidentlyResponseEvaluatorTool()
        stats_agent = ToolCallingAgent(
            tools=[stats_tool], 
            model=self.models['statistics'],  # 🔄 Dedicated model
            name="statistics_agent"
        )
        self.sub_agents['statistics'] = ManagedAgent(
            agent=stats_agent,
            name="statistics_evaluator",
            description="Evaluates response using statistical metrics."
        )
        
        # 4. Fact Checker Agent with dedicated model
        fact_checker_tool = FactCheckTool()
        fact_checker_agent = ToolCallingAgent(
            tools=[fact_checker_tool],
            model=self.models['fact_checker'],  # 🔄 Dedicated model
            name="fact_checker_agent"
        )
        self.sub_agents['fact_checker'] = ManagedAgent(
            agent=fact_checker_agent,
            name="fact_checker",
            description="Verifies factual accuracy of statements."
        )
        
        print(f"✅ Created {len(self.sub_agents)} sub-agents with dedicated models")
        
    def _create_manager_agent(self):
        """Create manager agent with its own dedicated model"""
        self.manager = CodeAgent(
            tools=[],
            model=self.models['manager'],  # 🔄 Dedicated manager model
            managed_agents=list(self.sub_agents.values()),
            name="evaluation_manager",
            additional_authorized_imports=["json", "pandas", "numpy", "matplotlib", "seaborn", "plotly"]
        )
        
        print("✅ Manager agent created with dedicated model")
    
    # def get_model_usage_summary(self):
    #     """Display which models are being used where"""
    #     summary = {
    #         "Manager (Coordination & Code Gen)": "models['manager']",
    #         "Clarity Evaluator": "models['clarity']", 
    #         "Pedagogy Evaluator": "models['pedagogy']",
    #         "Statistics Evaluator": "models['statistics']",
    #         "Fact Checker": "models['fact_checker']",
    #         "Total Separate Instances": len(self.models)
    #     }
        
    #     print("\n" + "="*50)
    #     print("MODEL INSTANCE DISTRIBUTION")
    #     print("="*50)
    #     for component, model in summary.items():
    #         print(f"{component:<30}: {model}")
    #     print("="*50)
        
    #     return summary


    def evaluate_response(self, question: str, answer: str) -> dict:
        """
        Main coordination method that manages all sub-agents and generates evaluation matrix
        """

        # Extract facts from answer for fact-checking
        facts = self._extract_facts_from_answer(answer)

        coordination_prompt = f"""
        You are the Manager Agent for a comprehensive response evaluation system.

        INPUTS:
        - Question: "{question}"
        - Answer: "{answer}"
        - API Key: "{self.api_key}"
        - Extracted Facts: {facts}

        TASK: Coordinate these 4 specialized sub-agents to evaluate the response:

        1. Call clarity_evaluator with:
           - question: "{question}"
           - answer: "{answer}"
           - gemini_api_key: "{self.api_key}"

        2. Call pedagogy_evaluator with:
           - question: "{question}"
           - answer: "{answer}"
           - gemini_api_key: "{self.api_key}"

        3. Call statistics_evaluator with:
           - question: "{question}"
           - answer: "{answer}"

        4. Call fact_checker with:
           - facts: {facts}

        After collecting all results, create a comprehensive evaluation matrix using pandas:

        ```
        import pandas as pd
        import numpy as np
        import logging

        # Configure logging to capture warnings about data inconsistencies
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("ManagerAgentDataValidation")

        # Collect results from all sub-agents
        results = {
            'clarity': clarity_results,
            'pedagogy': pedagogy_results, 
            'statistics': statistics_results,
            'fact_checking': fact_checker_results
        }

        # List of metrics we expect from each agent (optional: adjust according to your schema)
        expected_metrics = {
            'clarity': ['metric1', 'metric2'],  # e.g., replace with actual keys
            'pedagogy': ['Progression: Simple to Complex', 'Promotes Critical Thinking', 'Terms Clearly Defined', 'Effectiveness of Examples'],
            'statistics': ['Length', 'Sentence Count', 'Sentiment Score', 'Relevance Score', 'Hallucination Score'],
            'fact_checking': []  # e.g., can be ['Fact 1', 'Fact 2', ...] or left for all keys found
        }

        # Data structure for matrix
        matrix_data = {}

        for agent, result in results.items():
            if not isinstance(result, dict):
                logger.warning(f"Result from agent '{agent}' is not a dict. Setting all expected metrics to 0.")
                for metric in expected_metrics.get(agent, []):
                    matrix_data[f"{agent}_{metric}"] = 0
                continue

            for metric in expected_metrics.get(agent, result.keys()):
                score = result.get(metric, 0)
                # Check if score is numeric
                if score is None:
                    logger.warning(f"{agent}: Missing value for '{metric}', defaulting to 0.")
                    score = 0
                elif not isinstance(score, (int, float, np.integer, np.floating)):
                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        logger.warning(f"{agent}: Non-numeric value '{score}' for '{metric}', defaulting to 0.")
                        score = 0

                matrix_data[f"{agent}_{metric}"] = score

        # Convert to clean DataFrame for downstream visualization/plotting
        eval_matrix = pd.DataFrame([matrix_data])

        # Optional: Log matrix for review
        logger.info("Cleaned evaluation matrix:\n%s", eval_matrix.to_string())

        # Now safe to plot (heatmap, bar, etc.), as all values are numeric and missing handled

        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Heatmap of all scores
        scores_df = eval_matrix.T
        sns.heatmap(scores_df, annot=True, cmap='RdYlGn', ax=axes)
        axes.set_title('Evaluation Scores Heatmap')

        # 2. Bar chart by category
        category_scores = []
        categories = ['Clarity', 'Pedagogy', 'Statistics', 'Fact Checking']
        for cat in categories:
            cat_cols = [col for col in matrix_data.keys() if cat.lower() in col.lower()]
            if cat_cols:
                category_scores.append(np.mean([matrix_data[col] for col in cat_cols]))
            else:
                category_scores.append(0)

        axes[1].bar(categories, category_scores, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        axes[1].set_title('Average Scores by Category')
        axes[1].set_ylabel('Score')

        # 3. Radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        scores_norm = [score/5.0 for score in category_scores] + [category_scores/5.0]

        axes[1].remove()
        ax_radar = fig.add_subplot(2, 2, 3, projection='polar')
        ax_radar.plot(angles, scores_norm, 'bo-', linewidth=2)
        ax_radar.fill(angles, scores_norm, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_title('Response Quality Radar')

        # 4. Overall score gauge
        overall_score = np.mean(category_scores)
        axes[1][1].pie([overall_score, 5-overall_score],
                     labels=['Score', 'Remaining'],
                     autopct='%1.1f%%', startangle=90,
                     colors=['lightblue', 'lightgray'])
        axes[1][1].set_title(f'Overall Score: {{overall_score:.2f}}/5.0')

        plt.tight_layout()
        plt.savefig('evaluation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Return comprehensive results
        final_results = {{
            'individual_agent_results': results,
            'evaluation_matrix': eval_matrix.to_dict(),
            'category_scores': dict(zip(categories, category_scores)),
            'overall_score': overall_score,
            'matrix_saved': 'evaluation_matrix.png'
        }}

        print("\\n" + "="*60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Score: {{overall_score:.2f}}/5.0")
        print("\\nCategory Breakdown:")
        for cat, score in zip(categories, category_scores):
            print(f"  {{cat}}: {{score:.2f}}/5.0")
        print("\\nMatrix visualization saved as: evaluation_matrix.png")
        print("="*60)

        final_results
        ```

        Execute this code to coordinate all agents and generate the evaluation matrix with visualizations.
        """

        # Execute the coordination
        result = self.manager.run(coordination_prompt)
        return result

    def _extract_facts_from_answer(self, answer: str) -> list:
        """Extract factual statements from answer for fact-checking"""
        sentences = answer.split('.')
        facts = []

        # Simple heuristic for factual statements
        factual_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will', 'contains', 'includes']

        for sentence in sentences:
            sentence = sentence.strip()
            if (sentence and
                len(sentence) > 15 and
                any(indicator in sentence.lower() for indicator in factual_indicators)):
                facts.append(sentence)

        return facts[:4]  # Limit to 4 facts to avoid overwhelming fact-checker

# Usage Example
def run_multi_agent_evaluation():
    """Complete usage example"""

    # Initialize the manager system
    API_KEY = "your-gemini-api-key-here"
    manager = ResponseEvaluationManager(API_KEY)

    # Input data
    input_question = """You are an AI assistant designed to teach machine learning concepts.
    I am an engineering student with a basic understanding of mathematics.
    Explain what a neural network is in simple terms."""

    model_response = """Excellent question! A neural network is like a brain for a computer.
    It's made up of layers of interconnected 'neurons' that process information.
    When you show it lots of examples, like pictures of cats, it learns to recognize
    patterns and can then identify cats in new pictures it's never seen before.
    This process of learning from data is what makes it so powerful for tasks like
    image recognition, language translation, and even playing games."""

    print("Starting Multi-Agent Evaluation System...")
    print(f"Question: {input_question}")
    print(f"Response: {model_response}")
    print("\nCoordinating sub-agents...")

    # Run comprehensive evaluation
    evaluation_results = manager.evaluate_response(input_question, model_response)

    return evaluation_results

# Execute the system
if __name__ == "__main__":
    results = run_multi_agent_evaluation()
