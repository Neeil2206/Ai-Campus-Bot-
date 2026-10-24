from utils.local_llm import generate_response

def rewrite_query(query, chat_history):
    """
    Rewrites follow-up student queries to be standalone search queries using Ollama.
    If no history exists, returns the original query immediately.
    """
    if not chat_history:
        return query
        
    try:
        # Format the recent history for the rewriter model
        history_str = ""
        for msg in chat_history[-4:]:
            role = "Student" if msg["role"] == "user" else "CampusBot"
            history_str += f"{role}: {msg['content']}\n"
            
        system_prompt = (
            "You are an AI assistant helping to reformulate user questions for a document search engine.\n"
            "Given the conversation history and a new follow-up question, rewrite it as a standalone, "
            "complete search query. Do not answer the question; just output the search query and nothing else.\n"
            "If the follow-up question is self-contained or a greeting, return it exactly as is."
        )
        
        user_prompt = (
            f"Conversation History:\n{history_str}\n"
            f"Follow-up Question: {query}\n"
            f"Standalone Query:"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        rewritten = generate_response(messages, temperature=0.0)
        rewritten = rewritten.strip('"\'')
        return rewritten if rewritten else query
    except Exception as e:
        print(f"Error during query rewriting: {e}")
        return query  # Fallback to original query on failure

def generate_grounded_response(query, retrieved_chunks, chat_history):
    """
    Generates a response grounded in retrieved chunks using the Ollama Local LLM.
    Handles context formatting and grounding prompts to prevent hallucination.
    """
    # Check if the query is a simple greeting
    is_greeting = False
    cleaned_query = query.strip().lower().rstrip("?!.")
    greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "yo"}
    if cleaned_query in greetings or any(cleaned_query.startswith(g + " ") for g in greetings):
        is_greeting = True
        
    if not retrieved_chunks and not is_greeting:
        return "I couldn't find this information in the official college documents available to me. Please contact the college administration for confirmation."
        
    try:
        # Format context block
        context_str = ""
        if retrieved_chunks:
            for idx, chunk in enumerate(retrieved_chunks):
                context_str += f"--- CONTEXT BLOCK {idx+1} ---\n"
                context_str += f"Document: {chunk['document_name']}\n"
                context_str += f"Page: {chunk['page']}\n"
                context_str += f"Content:\n{chunk['text']}\n\n"
        else:
            context_str = "No relevant context chunks found in the database."

        system_instruction = (
            "You are CampusBot, an intelligent college admission assistant. Your objective is to help "
            "prospective students get accurate and conversational answers to college admission questions.\n\n"
            
            "CRITICAL GUIDELINES FOR GROUNDING:\n"
            "1. Answer user questions using ONLY the official information retrieved from the college knowledge base provided in the prompt.\n"
            "2. Do NOT invent or make up fees, eligibility criteria, deadlines, courses, scholarship amounts, hostel details, or any other institutional information.\n"
            "3. If the required information is NOT available in the retrieved documents, or if the retrieved documents do not contain the answer, you MUST respond EXACTLY with this sentence: \n"
            "\"I couldn't find this information in the official college documents available to me. Please contact the college administration for confirmation.\"\n"
            "Do NOT add any other explanation, recommendation, or guess after this sentence. It must be returned exactly.\n"
            "4. For general greetings (e.g., 'hello', 'hi', 'how are you'), respond warmly, introduce yourself, and state how you can help (eligibility, courses, fees, etc.), without needing to refer to document context.\n\n"
            
            "FORMATTING GUIDELINES:\n"
            "- Keep responses clear, concise, professional, and student-friendly.\n"
            "- Use markdown (bullet points, bold text) for readability.\n"
            "- You MUST cite the source document name and page number at the end of your response if you used context to answer it. "
            "Format the citation on a new line like this: 'Source: [Document Name], Page [Page Number].' "
            "If page information is not available, format it as: 'Source: [Document Name].' "
            "If multiple sources were used, list them clearly."
        )

        messages = [{"role": "system", "content": system_instruction}]
        
        # Append history
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        user_message_content = (
            f"Here is the retrieved context from the official database:\n\n{context_str}\n"
            f"Student Question: {query}\n\n"
            f"Answer the question using the retrieved context above. Remember the grounding rules."
        )
        messages.append({"role": "user", "content": user_message_content})
        
        answer = generate_response(messages, temperature=0.2)
        return answer
    except Exception as e:
        raise e
