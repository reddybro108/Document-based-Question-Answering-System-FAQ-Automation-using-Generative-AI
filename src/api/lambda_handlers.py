from src.core.qa_engine import DocumentQAEngine

qa_engine = DocumentQAEngine()

def lambda_handler(event, context):
    question = event.get("query")
    answer = qa_engine.ask(question)
    return {
        "statusCode": 200,
        "body": {
            "question": question,
            "answer": answer
        }
    }
