import sys
sys.path.append('app')

from history_retriever import rewrite_query
from hybrid_retriever import hybrid_retrieve
from reranker import rerank
from generator import build_prompt
from memory import ConversationMemory
from retrieve import retrieve

memory = ConversationMemory()
q = 'Do I need a valid CNIC?'
rewritten = rewrite_query(q, memory)
print('ORIGINAL:', q)
print('REWRITTEN:', rewritten)

faiss_results = retrieve(rewritten, top_k=5)
print('FAISS_COUNT:', len(faiss_results))
print('FAISS_SAMPLE:', [(r['vector_id'], getattr(r['chunk'], 'page_content', None)) for r in faiss_results[:3]])

hybrid = hybrid_retrieve(rewritten, faiss_top_k=5, bm25_top_k=5)
print('HYBRID_COUNT:', len(hybrid))
print('HYBRID_SAMPLE:', [(r['vector_id'], getattr(r['chunk'], 'page_content', None)) for r in hybrid[:3]])

reranked = rerank(rewritten, hybrid, top_n=3)
print('RERANKED_COUNT:', len(reranked))
print('RERANKED_SAMPLE:', [(r['vector_id'], r['score'], getattr(r['chunk'], 'page_content', None)) for r in reranked[:3]])

if reranked:
    prompt = build_prompt(q, reranked, memory)
    print('PROMPT_HAS_CONTEXT:', 'RETRIEVED CONTEXT' in prompt and 'CNIC' in prompt)
    print('PROMPT_CONTEXT_BLOCK:')
    print(prompt.split('RETRIEVED CONTEXT', 1)[1].split('CURRENT QUESTION', 1)[0][:2000])
