import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentQASystem:
   
    def __init__(self, document_text):
      
        if not document_text.strip():
            raise ValueError("Document text cannot be empty.")

        self.document_text = document_text
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize_and_clean)

        
        self.sentences = nltk.sent_tokenize(document_text)
        if not self.sentences:
            raise ValueError("Document does not contain any sentences.")
            
        
        self.doc_vectors = self.vectorizer.fit_transform(self.sentences)
        print("Document processed and TF-IDF model 'trained'.")

    def _tokenize_and_clean(self, text):
        
        tokens = nltk.word_tokenize(text.lower())
        
        cleaned_tokens = [
            token for token in tokens 
            if token not in string.punctuation and token not in self.stop_words
        ]
        return cleaned_tokens

    def answer_question(self, question, choices, top_n_sentences=3):
        
        if not question.strip():
            print("Warning: Question is empty.")
            return None, 0.0
        if not choices:
            print("Warning: No choices provided.")
            return None, 0.0

        
        question_vector = self.vectorizer.transform([question])

        
        similarities = cosine_similarity(question_vector, self.doc_vectors)
        
        
        sorted_sentence_indices = np.argsort(similarities[0])[::-1] 

        if not sorted_sentence_indices.size or similarities[0][sorted_sentence_indices[0]] == 0:
            print("Warning: No relevant sentences found for the question in the document.")
            return choices[0], 0.0

        
        relevant_sentence_indices = sorted_sentence_indices[:top_n_sentences]
        relevant_doc_vectors = self.doc_vectors[relevant_sentence_indices]
        
        

        best_choice = None
        max_similarity_score = -1.0

        
        for choice in choices:
            if not choice.strip():
                print(f"Warning: Skipping empty choice: '{choice}'")
                continue
            
            choice_vector = self.vectorizer.transform([choice])
            
            
            choice_similarities = cosine_similarity(choice_vector, relevant_doc_vectors)
            
            
            current_choice_max_similarity = np.max(choice_similarities) if choice_similarities.size > 0 else 0.0
            
            


            if current_choice_max_similarity > max_similarity_score:
                max_similarity_score = current_choice_max_similarity
                best_choice = choice
        
        if best_choice is None and choices: 
            print("Warning: Could not determine a best choice based on similarity. Defaulting to the first choice.")
            return choices[0], 0.0

        return best_choice, max_similarity_score

    def evaluate_performance(self, test_data):
        
        if not test_data:
            print("No test data provided for evaluation.")
            return 0.0

        correct_predictions = 0
        total_questions = len(test_data)

        print("\n--- Evaluating Performance ---")
        for i, item in enumerate(test_data):
            question = item['question']
            choices = item['choices']
            correct_answer_text = item['correct_answer_text']

            print(f"\nQ{i+1}: {question}")
            print(f"Choices: {choices}")
            print(f"Expected: {correct_answer_text}")

            predicted_answer, score = self.answer_question(question, choices)
            print(f"Predicted: {predicted_answer} (Score: {score:.4f})")

            if predicted_answer == correct_answer_text:
                correct_predictions += 1
                print("Result: CORRECT")
            else:
                print("Result: INCORRECT")
        
        accuracy = correct_predictions / total_questions if total_questions > 0 else 0.0
        print(f"\n--- Evaluation Summary ---")
        print(f"Total Questions: {total_questions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
        print("--------------------------")
        return accuracy


if __name__ == "__main__":
    

    with open("Whispers of Adventure.txt", "r", encoding="utf-8") as f:
        sample_document_content = f.read()
    
    

    try:
        
        qa_system = DocumentQASystem(sample_document_content)

        
        question1 = "Which planet is known as the Red Planet?"
        choices1 = ["Earth", "Mars", "Jupiter", "Venus"]
        
        predicted_answer1, score1 = qa_system.answer_question(question1, choices1)
        
        print(f"\n--- Question 1 ---")
        print(f"Question: {question1}")
        print(f"Choices: {choices1}")
        if predicted_answer1 is not None:
            print(f"Predicted Answer: {predicted_answer1} (Confidence Score: {score1:.4f})")
        else:
            print("Could not determine an answer.")

        question2 = "What are Jupiter and Saturn primarily composed of?"
        choices2 = ["Rock and metal", "Ice and water", "Hydrogen and helium", "Ammonia and methane"]
        predicted_answer2, score2 = qa_system.answer_question(question2, choices2)

        print(f"\n--- Question 2 ---")
        print(f"Question: {question2}")
        print(f"Choices: {choices2}")
        if predicted_answer2 is not None:
            print(f"Predicted Answer: {predicted_answer2} (Confidence Score: {score2:.4f})")
        else:
            print("Could not determine an answer.")

        question3 = "Which is the largest planet?"
        choices3 = ["Earth", "Mars", "Jupiter", "Saturn"]
        predicted_answer3, score3 = qa_system.answer_question(question3, choices3)
        
        print(f"\n--- Question 3 ---")
        print(f"Question: {question3}")
        print(f"Choices: {choices3}")
        if predicted_answer3 is not None:
            print(f"Predicted Answer: {predicted_answer3} (Confidence Score: {score3:.4f})")
        else:
            print("Could not determine an answer.")

        
        sample_test_data = [
            # {
            #     "question": "What the Pip suddenly chirped?",
            #     "choices": ["Stone",   "Fire", "Car", "A stamp"],
            #     "correct_answer_text": "A stamp"
            # },
            # {
            #     "question": "What was missing from the Moonstone Glade?",
            #     "choices": ["A tree", "A flower", "The moonstone", "A stream"],
            #     "correct_answer_text": "The moonstone"
            # },
            # {
            #     "question": "Who delivered the urgent message to Finn?",
            #     "choices": [" Benny the badger", "Luna the owl", "Pip the sparrow", "The honey bear"],
            #     "correct_answer_text": "Pip the sparrow" 
            # },
           

            # {
            #     "question": "What did the sprites ask Finn and his friends to do?",
            #     "choices": ["Dance with them", "Solve their riddles", "Find more moonstones", "Build a treehouse"],
            #     "correct_answer_text": "Solve their riddles"
            # },
            # {
            #     "question": "What did Finn and his friends do to celebrate after finding the moonstone?",
            #     "choices": ["Go on another adventure", "Have a party in the glade", "Build a new den", "Make a treasure map"],
            #     "correct_answer_text": "Have a party in the glade"
            # }
            {
            "question": "What is the name of the young explorer in the story?",
            "choices": ["Lila", "Malgor", "Zephyr", "Willow"],
            "correct_answer_text": "Lila"
            },
            {
            "question": "Where does Lila find the crystal orb?",
            "choices": ["In her backpack", "At the village square", "On a pedestal in a grand chamber", "Inside a tree"],
            "correct_answer_text": "On a pedestal in a grand chamber"
            },
            {
            "question": "What is Zephyr's role in the ruins?",
            "choices": ["A thief", "The Guardian", "Lila's father", "A villager"],
            "correct_answer_text": "The Guardian"
            }



        ]
        qa_system.evaluate_performance(sample_test_data)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

