from flask import Flask, request, jsonify
import os
import json
import traceback
from recommendation_engine import RecommendationEngine

app = Flask(__name__)

# Initialize the recommendation engine
engine = RecommendationEngine("processed_books.csv.zip")



# Mapping functions for converting natural language to parameters
def map_rating_level(rating_level):
    """Map rating level descriptions to numerical values."""
    if rating_level in ["high rating", "good rating", "excellent", "high", "good"]:
        return 4.0
    elif rating_level in ["average", "medium", "decent"]:
        return 3.0
    return 0  # Default: no rating filter

def map_length_level(length_level):
    """Map length level descriptions to page count values."""
    if length_level in ["short", "brief", "quick read"]:
        return 300
    elif length_level in ["medium", "average length", "moderate"]:
        return (300, 500)  # Returns a tuple for range
    elif length_level in ["long", "lengthy", "epic"]:
        return 500
    return None  # Default: no page limit

def get_book_from_context(contexts):
    """Extract book title from conversation context."""
    for context in contexts:
        if 'parameters' in context and 'book_title' in context['parameters']:
            return context['parameters']['book_title']
    return None

@app.route('/', methods=['GET'])
def index():
    """Basic endpoint to verify the API is running."""
    return jsonify({
        'status': 'online',
        'message': 'Book Recommendation API is running'
    })

def build_context_name(context_type, request_json=None):
    """
    Build a proper context name with project ID and session ID.
    
    Args:
        context_type: The type of context (e.g. 'awaiting_genre')
        request_json: The JSON request object to extract session information
        
    Returns:
        Full context name for Dialogflow
    """
    PROJECT_ID = "group26-finalproject-n9iq"
    SESSION_ID = "SESSION_ID"  # Default value
    
    # If request JSON is provided, extract session ID
    if request_json:
        contexts = request_json.get('queryResult', {}).get('outputContexts', [])
        if contexts:
            context_name = contexts[0].get('name', '')
            parts = context_name.split('/')
            if len(parts) >= 5:
                SESSION_ID = parts[4]  # Session ID is the 5th part in the context path
    
    return f"projects/{PROJECT_ID}/agent/sessions/{SESSION_ID}/contexts/{context_type}"

    
@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle webhook requests from Dialogflow."""
    req = request.get_json(silent=True, force=True)
    
    # Log the incoming request for debugging
    print("Request from Dialogflow:")
    print(json.dumps(req, indent=2))
    
    try:
        intent = req['queryResult']['intent']['displayName']
        parameters = req['queryResult']['parameters']
        
        print(f"Processing intent: {intent}")
        
        # Route to appropriate handler based on intent
        if intent in ['Collect_Book_Genre', 'Collect_Book_Style', 'Collect_Rating_Preference', 'Collect_Book_Length']:
            return handle_collect_preferences(intent, parameters)
        elif intent == 'Request_Book_Details':
            return handle_book_details(parameters)
        elif intent == 'Request_Similar_Books':
            return handle_similar_books(req['queryResult'])

        elif intent == 'Request_New_Conversation - yes':
            #User want to new search
            return jsonify({
                'fulfillmentText': 'Great! Let\'s find you some new books. What kind of books are you interested in?',
                'followupEventInput': {
                    'name': 'WELCOME_EVENT',
                    'languageCode': 'en-US'
                }
            })
        elif intent == 'Goodbye':
            #User dont want to new search
            return jsonify({
                'fulfillmentText': 'Thank you for using our book recommendation service! If you need new recommendations in the future, I\'m always here to help.'
            })
        else:
            return jsonify({
                'fulfillmentText': 'I\'m not sure how to help with that. Would you like a book recommendation?'
            })
    
    except Exception as e:
        print(f"Error processing request: {e}")
        # Print full stack trace for debugging
        traceback.print_exc()
        return jsonify({
            'fulfillmentText': 'Sorry, I encountered an error processing your request. Please try again.'
        })
def handle_collect_preferences(intent, parameters):
    """
    Collect book preferences from the conversation context:
    Check sequentially whether the book type, style, rating, and page count have been collected.
    If any item is missing, prompt the user for that information; otherwise, call the recommendation interface.
    """
    contexts = request.get_json(silent=True, force=True).get('queryResult', {}).get('outputContexts', [])
    
    # Helper function to find context value
    def get_context_parameter(context_name, param_name):
        for context in contexts:
            if context_name in context.get('name', ''):
                return context.get('parameters', {}).get(param_name, '')
        return ''
    
    # Extract all related parameters
    genre = parameters.get('genre', '') or get_context_parameter('awaiting_genre', 'genre')
    style = parameters.get('style', '') or get_context_parameter('awaiting_style', 'style')
    rating_level = parameters.get('rating_level', '') or get_context_parameter('awaiting_rating', 'rating_level')
    length_level = parameters.get('length_level', '') or get_context_parameter('awaiting_length', 'length_level')

    # Prepare output contexts to maintain conversation state
    output_contexts = []
    
    # Determine the next step in preference collection
    if not genre:
        return jsonify({
            'fulfillmentText': "Hello! I am the book recommendation assistant. I can recommend books for you based on your preferences. May I ask what type of books you are interested in?",
            'outputContexts': [{
                'name': build_context_name('awaiting_genre', req),
                'lifespanCount': 5,
                'parameters': {}
            }]
        })
    
    elif not style:
        return jsonify({
            'fulfillmentText': "What kind of book style do you prefer? For example, do you like humorous, adventurous, or serious books?",
            'outputContexts': [{
                'name': build_context_name('awaiting_style', req),
                'lifespanCount': 5,
                'parameters': {
                    'genre': genre
                }
            }]
        })
    
    elif not rating_level:
        return jsonify({
            'fulfillmentText': "Do you have any rating requirements for the books? For example, should I only recommend books with ratings above 4.0?",
            'outputContexts': [{
                'name': build_context_name('awaiting_rating', req),
                'lifespanCount': 5,
                'parameters': {
                    'genre': genre,
                    'style': style
                }
            }]
        })
    
    elif not length_level:
        return jsonify({
            'fulfillmentText': "Do you have any page count requirements for the books? For example, would you like me to recommend books with fewer than 300 pages?",
            'outputContexts': [{
                'name': build_context_name('awaiting_length', req),
                'lifespanCount': 5,
                'parameters': {
                    'genre': genre,
                    'style': style,
                    'rating_level': rating_level
                }
            }]
        })
    
    else:
        # Once all parameters are collected, call the recommendation logic
        return handle_book_recommendation(parameters)

def handle_book_recommendation(parameters):
    """Process book recommendation request based on collected parameters."""
    # Extract and map parameters
    genre = parameters.get('genre', '')
    style = parameters.get('style', '')
    rating_level = parameters.get('rating_level', '')
    length_level = parameters.get('length_level', '')
    
    # Map to numerical values
    min_rating = map_rating_level(rating_level)
    max_pages = None
    
    length_value = map_length_level(length_level)
    if isinstance(length_value, tuple):
        max_pages = length_value[1]  # Use the upper bound
    else:
        max_pages = length_value
    
    # Get recommendations using ensemble method
    recommendations = engine.ensemble_recommendations(
        genre=genre, 
        style=style, 
        min_rating=min_rating, 
        max_pages=max_pages, 
        top_n=3
    )
    
    if recommendations.empty:
        return jsonify({
            'fulfillmentText': 'I couldn\'t find any books matching your preferences. Would you like to try with different criteria?'
        })

    preferences = []
    if genre:
        preferences.append(f"genre: {genre}")
    if style:
        preferences.append(f"style: {style}")
    if rating_level:
        preferences.append(f"rating: {rating_level}")
    if length_level:
        preferences.append(f"length: {length_level}")
    
    preferences_text = ", ".join(preferences) if preferences else "your preferences"
    
    
    # Format response
    response_text = "Based on your preferences({preferences_text}), I recommend these books:\n\n"
    
    for _, book in recommendations.iterrows():
        response_text += f"• {book['book_title']} – by {book['book_authors']}, "
        response_text += f"rating: {book['book_rating']:.1f}, {book['book_pages']} pages\n"
    
    response_text += "\nWould you like to know more details about any of these books?"
    
    return jsonify({
        'fulfillmentText': response_text
    })

def handle_book_details(parameters):
    """Handle request for detailed information about a specific book."""
    req = request.get_json(silent=True, force=True)
    book_title = parameters.get('book_title', '')
    
    # If no book title, indicate what to do next
    if not book_title:
        return jsonify({
            'fulfillmentText': 'Which book would you like to know more about?',
            'outputContexts': [{
                'name': build_context_name('awaiting_details', req),
                'lifespanCount': 5
            }]
        })
    
    book_details = engine.get_book_details(book_title)
    
    if not book_details:
        return jsonify({
            'fulfillmentText': f'I couldn\'t find details for a book titled "{book_title}". Could you check the spelling or try another book?',
            'outputContexts': [{
                'name': build_context_name('awaiting_details', req),
                'lifespanCount': 5,
                'parameters': {
                    'book_title': book_title
                }
            }]
        })
    
    # Format detailed response
    response_text = f"Here are the details about '{book_details['title']}':\n\n"
    response_text += f"【Description】 {book_details['description']}\n\n"
    response_text += f"【Author】 {book_details['author']}\n"
    response_text += f"【Genre】 {book_details['genres']}\n"
    response_text += f"【Rating】 {book_details['rating']:.1f} (based on {book_details['rating_count']} reviews)\n"
    response_text += f"【Pages】 {book_details['pages']}\n\n"
    response_text += "Would you like to find books similar to this one?"
    
    return jsonify({
        'fulfillmentText': response_text,
        'outputContexts': [{
            'name': build_context_name('has_book_context', req),
            'lifespanCount': 5,
            'parameters': {
                'book_title': book_details['title']
            }
        }]
    })

def handle_similar_books(query_result):
    """Handle request for books similar to a specific title."""
    req = request.get_json(silent=True, force=True)
    parameters = query_result.get('parameters', {})
    book_title = parameters.get('book_title', '')
    
    # If no book title in parameters, try to get from context
    if not book_title:
        book_title = get_book_from_context(query_result.get('outputContexts', []))
    
    if not book_title:
        return jsonify({
            'fulfillmentText': 'Which book would you like to find similar titles for?',
            'outputContexts': [{
                'name': build_context_name('has_book_context', req),
                'lifespanCount': 5
            }]
        })
    
    similar_books = engine.find_similar_books_knn(book_title, n=3)
    
    if similar_books.empty:
        return jsonify({
            'fulfillmentText': f'I couldn\'t find any books similar to "{book_title}". Would you like to try with a different book?',
            'outputContexts': [{
                'name': build_context_name('has_book_context', req),
                'lifespanCount': 5,
                'parameters': {
                    'book_title': book_title
                }
            }]
        })
    
    # Format response
    response_text = f"Here are some books similar to '{book_title}':\n\n"
    
    for _, book in similar_books.iterrows():
        response_text += f"• {book['book_title']} – by {book['book_authors']}, "
        response_text += f"rating: {book['book_rating']:.1f}\n"
    
    response_text += "\nWould you like to start a new recommendation search?"
    
    return jsonify({
        'fulfillmentText': response_text,
        'outputContexts': [{
            'name': build_context_name('has_book_context', req),
            'lifespanCount': 5,
            'parameters': {
                'book_title': book_title
            }
        }]
    })
    
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
