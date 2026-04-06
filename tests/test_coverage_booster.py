import unittest
from unittest.mock import MagicMock, patch
import os
import networkx as nx

from ingestion.clone_repo import clone_repository
from reasoning.architecture_analyzer import ArchitectureAnalyzer
from reasoning.query_planner import AgenticQueryPlanner
from reasoning.answer_generator import AnswerGenerator
from observability.monitoring import configure_structured_logging


class TestCoverageBooster(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("indexing.cache_manager.LocalCacheManager.get", return_value=None)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_clone_repository_local(self):
        # Test local directory branch
        path = os.path.dirname(os.path.abspath(__file__))
        res = clone_repository(path)
        self.assertEqual(res, path)

    @patch("ingestion.clone_repo.Repo")
    def test_clone_repository_existing(self, mock_repo_cls):
        mock_repo = MagicMock()
        mock_repo_cls.return_value = mock_repo
        
        with patch("os.path.exists", return_value=True):
            res = clone_repository("https://github.com/user/my-repo", "/tmp/target")
            self.assertEqual(res, "/tmp/target")
            mock_repo.remotes.origin.pull.assert_called_once()

    def test_architecture_analyzer_sequence_chart(self):
        analyzer = MagicMock()
        kg = MagicMock()
        g = nx.DiGraph()
        g.add_node("main.py.main_func")
        g.add_node("main.py.other_func")
        g.add_edge("main.py.main_func", "main.py.other_func", type="calls")
        kg.graph = g
        analyzer._kg = kg

        arch = ArchitectureAnalyzer(analyzer)
        chart = arch.generate_sequence_chart("main_func")
        self.assertIn("sequenceDiagram", chart)
        self.assertIn("main_func ->> other_func: call", chart)

        # Test function not found
        chart_not_found = arch.generate_sequence_chart("nonexistent_func")
        self.assertIn("FunctionNotFound", chart_not_found)

    def test_query_planner_execute_all_tools(self):
        analyzer = MagicMock()
        analyzer.find_references.return_value = "references_val"
        analyzer.find_implementations.return_value = "implementations_val"
        analyzer.find_inheritance.return_value = "inheritance_val"
        analyzer.find_dependency_chains.return_value = "dependency_chains_val"
        analyzer.ask_question.return_value = {"answer": "synthesized"}

        planner = AgenticQueryPlanner(analyzer)
        plan = [
            {"tool": "find_references", "symbol_name": "my_symbol"},
            {"tool": "find_implementations", "class_name": "my_class"},
            {"tool": "find_inheritance", "class_name": "my_class"},
            {"tool": "find_dependency_chains", "file_path": "my_file.py"},
            {"tool": "other_unknown", "query": "what is this?"}
        ]
        
        res = planner.execute_plan(plan)
        self.assertEqual(res, {"answer": "synthesized"})

    @patch("google.generativeai.GenerativeModel")
    def test_query_planner_create_plan_gemini(self, mock_gemini_model):
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value.text = '[{"tool": "ask_question", "query": "hello"}]'
        mock_gemini_model.return_value = mock_instance

        planner = MagicMock()
        p = AgenticQueryPlanner(planner)
        
        from config import config
        old_provider = config.llm_provider
        config.llm_provider = "gemini"
        try:
            plan = p.create_plan("hello")
            self.assertEqual(plan[0]["tool"], "ask_question")
        finally:
            config.llm_provider = old_provider

    def test_query_planner_create_plan_openai(self):
        from config import config
        old_provider = config.llm_provider
        config.llm_provider = "openai"
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-mock-key-value-here"
        try:
            planner = MagicMock()
            p = AgenticQueryPlanner(planner)
            p._openai_client = MagicMock()
            mock_res = MagicMock()
            mock_res.choices = [MagicMock()]
            mock_res.choices[0].message.content = '[{"tool": "ask_question", "query": "hello"}]'
            p._openai_client.chat.completions.create.return_value = mock_res
            plan = p.create_plan("hello")
            self.assertEqual(plan[0]["tool"], "ask_question")
        finally:
            config.llm_provider = old_provider
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_configure_structured_logging(self):
        configure_structured_logging()

    def test_answer_generator_fallback(self):
        retriever = MagicMock()
        retriever.retrieve_with_context.return_value = "context"
        retriever.retrieve.return_value = []
        gen = AnswerGenerator(retriever, repo_name="test_repo")
        
        # Test Gemini empty text response
        if gen._use_gemini:
            gen._gemini_model.generate_content = MagicMock(return_value=None)
            res = gen.generate_answer("hello")
            self.assertEqual(res["answer"], "")

    def test_answer_generator_openai(self):
        retriever = MagicMock()
        retriever.retrieve_with_context.return_value = "context"
        retriever.retrieve.return_value = []
        
        from config import config
        old_provider = config.llm_provider
        config.llm_provider = "openai"
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-mock-key-value-here"
        try:
            gen = AnswerGenerator(retriever, repo_name="test_repo")
            gen._openai_client = MagicMock()
            
            mock_res = MagicMock()
            mock_res.choices = [MagicMock()]
            mock_res.choices[0].message.content = "openai response"
            gen._openai_client.chat.completions.create.return_value = mock_res
            
            res = gen.generate_answer("hello-openai-booster")
            self.assertEqual(res["answer"], "openai response")
            
            # Stream
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = "token"
            gen._openai_client.chat.completions.create.return_value = [mock_chunk]
            stream_res = list(gen.generate_answer_stream("hello-openai-booster-stream"))
            self.assertEqual(stream_res[1]["text"], "token")
        finally:
            config.llm_provider = old_provider
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]



    def test_openai_embedder(self):
        from indexing.embedder import OpenAIEmbedder
        from ingestion.chunk_code import CodeChunk
        import os
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-mock-key"
        try:
            embedder = OpenAIEmbedder()
            embedder.client = MagicMock()
            
            mock_res = MagicMock()
            mock_item = MagicMock()
            mock_item.embedding = [0.1, 0.2]
            mock_res.data = [mock_item]
            embedder.client.embeddings.create.return_value = mock_res
            
            vecs = embedder.embed_texts(["hello"])
            self.assertEqual(vecs, [[0.1, 0.2]])

            # Test embed_chunks
            chunk = CodeChunk(
                id="c1",
                file_path="f.py",
                content="code",
                start_line=1,
                end_line=10,
                symbol_name="sym"
            )
            vecs_chunks = embedder.embed_chunks([chunk])
            self.assertEqual(vecs_chunks, [[0.1, 0.2]])
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    @patch("time.sleep")
    def test_gemini_embedder(self, mock_sleep):
        from indexing.embedder import GeminiEmbedder
        embedder = GeminiEmbedder(api_key="mock-gemini-key")
        embedder._genai = MagicMock()
        
        # Test batch embed with attributes
        mock_result = MagicMock()
        mock_result.embedding = [[0.3, 0.4]]
        embedder._genai.embed_content.return_value = mock_result
        vecs = embedder.embed_texts(["hello"])
        self.assertEqual(vecs, [[0.3, 0.4]])
        
        # Test batch embed fallback list dict
        embedder._genai.embed_content.return_value = {"embeddings": [[0.5, 0.6]]}
        vecs2 = embedder.embed_texts(["world"])
        self.assertEqual(vecs2, [[0.5, 0.6]])

        # Test retry on 429 and batching (21 items to exceed BATCH_SIZE=20)
        # First call raises 429, second call succeeds
        call_count = 0
        def mock_embed(model, content, task_type):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Resource exhausted (429)")
            return {"embeddings": [[0.7] * 768] * len(content)}

        embedder._genai.embed_content.side_effect = mock_embed
        texts = ["text"] * 21
        vecs3 = embedder.embed_texts(texts)
        self.assertEqual(len(vecs3), 21)
        self.assertTrue(mock_sleep.called)

    @patch("ingestion.clone_repo.Repo")
    def test_clone_repository_pull_fail(self, mock_repo_cls):
        from ingestion.clone_repo import clone_repository
        
        # Test pulling fails
        mock_repo = MagicMock()
        mock_repo.remotes.origin.pull.side_effect = Exception("pull error")
        mock_repo_cls.return_value = mock_repo
        
        import tempfile, shutil, os
        tmp = tempfile.mkdtemp()
        try:
            res = clone_repository("https://github.com/mock/repo", target_dir=tmp)
            self.assertEqual(res, tmp)
        finally:
            shutil.rmtree(tmp)


    def test_answer_generator_gemini_success(self):
        retriever = MagicMock()
        retriever.retrieve_with_context.return_value = "context"
        retriever.retrieve.return_value = []
        
        from config import config
        old_provider = config.llm_provider
        config.llm_provider = "gemini"
        old_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "sk-mock-gemini-key-value"
        try:
            gen = AnswerGenerator(retriever, repo_name="test_repo")
            gen.conversation_manager.add_message("test-conv", "user", "hi")
            
            # Normal
            mock_res = MagicMock()
            mock_res.text = "gemini success answer"
            gen._gemini_model.generate_content = MagicMock(return_value=mock_res)
            res = gen.generate_answer("hello-gemini-success", conversation_id="test-conv")
            self.assertEqual(res["answer"], "gemini success answer")
            
            # Stream
            mock_chunk = MagicMock()
            mock_chunk.text = "gemini token"
            gen._gemini_model.generate_content = MagicMock(return_value=[mock_chunk])
            stream_res = list(gen.generate_answer_stream("hello-gemini-success-stream", conversation_id="test-conv"))
            self.assertEqual(stream_res[1]["text"], "gemini token")
        finally:
            config.llm_provider = old_provider
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]


if __name__ == "__main__":
    unittest.main()
