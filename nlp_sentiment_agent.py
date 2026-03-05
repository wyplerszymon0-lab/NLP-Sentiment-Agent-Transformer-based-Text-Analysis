from transformers import pipeline

class SentimentAgent:
    def __init__(self):
        self.analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def analyze(self, text):
        result = self.analyzer(text)[0]
        return result

if __name__ == "__main__":
    agent = SentimentAgent()
    
    test_texts = [
        "This new AI system is absolutely revolutionary and efficient!",
        "The project failed due to poor architecture and lack of communication.",
        "It is a simple script that gets the job done."
    ]

    for text in test_texts:
        res = agent.analyze(text)
        print(f"Text: {text}")
        print(f"Label: {res['label']} | Score: {res['score']:.4f}\n")
