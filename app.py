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
    if not rating_level:
        return 0
    
    rating_level = rating_level.lower()
    
    if any(term in rating_level for term in ["high rating", "good rating", "excellent", "high", "good", "best", "top"]):
        return 4.0
    elif any(term in rating_level for term in ["average", "medium", "decent", "ok", "okay"]):
        return 3.0
    elif any(term in rating_level for term in ["any", "no preference", "not important"]):
        return 0
    
    # Try to extract a numerical value if present
    import re
    numbers = re.findall(r'\d+(?:\.\d+)?', rating_level)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass
    
    return 0  # Default: no rating filter

def map_length_level(length_level):
    """Map length level descriptions to page count values.
    Returns a single integer representing the maximum page count.
    """
    if not length_level:
        return None
    
    # Convert to lowercase for case-insensitive matching
    if isinstance(length_level, str):
        length_level = length_level.lower()
    else:
        return None
    
    # Map common phrases to page counts
    if any(term in length_level for term in ["short", "brief", "quick read", "small"]):
        return 300  # Books with 300 pages or less
    elif any(term in length_level for term in ["medium", "average length", "moderate", "normal"]):
        return 500  # Books with 500 pages or less
    elif any(term in length_level for term in ["long", "lengthy", "epic", "big"]):
        return 800  # Books with 800 pages or less
    elif any(term in length_level for term in ["any", "no preference", "not important"]):
        return None  # No page limit
    
    # Try to extract a numerical value if present
    import re
    numbers = re.findall(r'\d+', length_level)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            pass
    
    return None
    
def get_all_context_parameters(req):
    """
    Merge parameters from all output contexts to carry over information across multi-turn conversations.
    """
    contexts = req.get('queryResult', {}).get('outputContexts', [])
    params = {}
    for context in contexts:
        params.update(context.get('parameters', {}))
    return params

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
    
    if request_json:
        contexts = request_json.get('queryResult', {}).get('outputContexts', [])
        if contexts:
            context_name = contexts[0].get('name', '')
            parts = context_name.split('/')
            if len(parts) >= 5:
                SESSION_ID = parts[4]
    
    return f"projects/{PROJECT_ID}/agent/sessions/{SESSION_ID}/contexts/{context_type}"

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle webhook requests from Dialogflow."""
    req = request.get_json(silent=True, force=True)
    
    print("Request from Dialogflow:")
    print(json.dumps(req, indent=2))
    
    try:
        intent = req['queryResult']['intent']['displayName']
        parameters = req['queryResult']['parameters']
        
        print(f"Processing intent: {intent}")
        
        if intent in ['Collect_Book_Genre', 'Collect_Book_Style', 'Collect_Rating_Preference', 'Collect_Book_Length']:
            return handle_collect_preferences(intent, parameters, req)
        elif intent == 'Request_Book_Details':
            return handle_book_details(parameters, req)
        elif intent == 'Request_Similar_Books':
            return handle_similar_books(req['queryResult'], req)

        elif intent == 'Request_New_Conversation - yes' or 'yes' in parameters.get('input', '').lower():
            # Clear all existing contexts by setting lifespan to 0
            output_contexts = []
            for context in req.get('queryResult', {}).get('outputContexts', []):
                context_name = context.get('name', '')
                output_contexts.append({
                    'name': context_name,
                    'lifespanCount': 0  # Setting to 0 clears the context
                })
            
            # Add a new clean welcome context
            output_contexts.append({
                'name': build_context_name('welcome_state', req),
                'lifespanCount': 5,
                'parameters': {}
            })
            
            return jsonify({
                'fulfillmentText': 'Great! Let\'s find you some new books. What kind of books are you interested in?',
                'outputContexts': output_contexts,
                'followupEventInput': {
                    'name': 'WELCOME_EVENT',
                    'languageCode': 'en-US'
                }
            })
        elif intent == 'Goodbye' or intent == 'Request_New_Conversation - no' or ('bye' in parameters.get('input', '').lower()):
            # Clear all existing contexts by setting lifespan to 0
            output_contexts = []
            for context in req.get('queryResult', {}).get('outputContexts', []):
                context_name = context.get('name', '')
                output_contexts.append({
                    'name': context_name,
                    'lifespanCount': 0  # Setting to 0 clears the context
                })
            
            return jsonify({
                'fulfillmentText': 'Thank you for using our book recommendation service! If you need new recommendations in the future, I\'m always here to help.',
                'outputContexts': output_contexts
            })
        else:
            return jsonify({
                'fulfillmentText': 'I\'m not sure how to help with that. Would you like a book recommendation?'
            })
    
    except Exception as e:
        print(f"Error processing request: {e}")
        traceback.print_exc()
        return jsonify({
            'fulfillmentText': 'Sorry, I encountered an error processing your request. Please try again.'
        })

def handle_collect_preferences(intent, parameters, req):
    """
    Collect book preference info:
    Check if genre, style, rating_level, and length_level are all present.
    If any are missing, ask the appropriate follow-up question, while storing the collected parameters in context.
    Once all are collected, proceed to generate recommendations.
    """
    all_context_params = get_all_context_parameters(req)
    
    genre = parameters.get('genre', '') or all_context_params.get('genre', '')
    style = parameters.get('style', '') or all_context_params.get('style', '')
    rating_level = parameters.get('rating_level', '') or all_context_params.get('rating_level', '')
    length_level = parameters.get('length_level', '') or all_context_params.get('length_level', '')
    
    if not genre:
        return jsonify({
            'fulfillmentText': "What genre of books do you enjoy?",
            'outputContexts': [{
                'name': build_context_name('awaiting_genre', req),
                'lifespanCount': 5,
                'parameters': {}
            }]
        })
    
    elif not style:
        return jsonify({
            'fulfillmentText': "What kind of style are you looking for? For example, do you prefer humorous, adventurous, or serious books?",
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
            'fulfillmentText': "Do you have a rating preference? For example, only recommend books rated 4.0 and above?",
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
            'fulfillmentText': "Do you have a page length preference? For example, books under 300 pages?",
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
        return handle_book_recommendation({
            'genre': genre,
            'style': style,
            'rating_level': rating_level,
            'length_level': length_level
        })

def handle_book_recommendation(parameters):
    """Call the recommendation engine and return recommended books based on collected preferences."""
    genre = parameters.get('genre', '')
    style = parameters.get('style', '')
    rating_level = parameters.get('rating_level', '')
    length_level = parameters.get('length_level', '')
    
    print(f"Raw parameters: genre={genre}, style={style}, rating_level={rating_level}, length_level={length_level}")
    
    min_rating = map_rating_level(rating_level)
    max_pages = map_length_level(length_level)  # This now consistently returns a single integer or None
    
    print(f"Mapped parameters: min_rating={min_rating}, max_pages={max_pages}")
    
    recommendations = engine.ensemble_recommendations(
        genre=genre, 
        style=style, 
        min_rating=min_rating, 
        max_pages=max_pages, 
        top_n=3
    )
    
    if recommendations.empty:
        return jsonify({
            'fulfillmentText': 'Sorry, no books matched your preferences. Would you like to change the criteria?'
        })
    
    preferences_text = f"Genre: {genre}, Style: {style}, Rating: {rating_level}, Length: {length_level}"
    response_text = f"Based on your preferences ({preferences_text}), I recommend the following books:\n\n"
    
    for _, book in recommendations.iterrows():
        response_text += f"• {book['book_title']} – Author: {book['book_authors']}, Rating: {book['book_rating']:.1f}, Pages: {book['book_pages']}\n"
    
    response_text += "\nWould you like to know more about any of these books?"
    
    return jsonify({
        'fulfillmentText': response_text,
        'outputContexts': [{
            'name': build_context_name('has_preferences', None),
            'lifespanCount': 10,
            'parameters': parameters
        }]
    })
    
def handle_book_details(parameters, req):
    """Return detailed information about a specified book with improved error handling."""
    book_title = parameters.get('book_title', '')
    
    print(f"Received book detail request for: '{book_title}'")
    
    if not book_title:
        return jsonify({
            'fulfillmentText': 'Which book would you like details about?',
            'outputContexts': [{
                'name': build_context_name('awaiting_details', req),
                'lifespanCount': 5
            }]
        })
    
    book_details = engine.get_book_details(book_title)
    
    if not book_details:
        # Try to get the book from the recommendation context
        all_context_params = get_all_context_parameters(req)
        if 'recommendations' in all_context_params:
            # This assumes you've stored recommendations in the context
            for rec_book in all_context_params.get('recommendations', []):
                if any(term in rec_book['book_title'].lower() for term in book_title.lower().split()):
                    print(f"Found book in recommendation context: {rec_book['book_title']}")
                    book_details = engine.get_book_details(rec_book['book_title'])
                    break
    
    if not book_details:
        return jsonify({
            'fulfillmentText': f'I couldn\'t find a book with the title "{book_title}". Could you check the spelling or try a different book?',
            'outputContexts': [{
                'name': build_context_name('awaiting_details', req),
                'lifespanCount': 5,
                'parameters': {
                    'book_title': book_title
                }
            }]
        })
    
    response_text = f"Details for \"{book_details['title']}\":\n\n"
    response_text += f"【Description】 {book_details['description']}\n\n"
    response_text += f"【Author】 {book_details['author']}\n"
    response_text += f"【Genre】 {book_details['genres']}\n"
    response_text += f"【Rating】 {book_details['rating']:.1f} (based on {book_details['rating_count']} reviews)\n"
    response_text += f"【Pages】 {book_details['pages']}\n\n"
    response_text += "Would you like recommendations for similar books?"
    
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

def handle_similar_books(query_result, req):
    """Return book recommendations similar to the given title."""
    parameters = query_result.get('parameters', {})
    book_title = parameters.get('book_title', '')
    
    if not book_title:
        all_contexts = req.get('queryResult', {}).get('outputContexts', [])
        book_title = get_book_from_context(all_contexts)
    
    if not book_title:
        return jsonify({
            'fulfillmentText': 'Which book would you like similar recommendations for?',
            'outputContexts': [{
                'name': build_context_name('has_book_context', req),
                'lifespanCount': 5
            }]
        })
    
    similar_books = engine.find_similar_books_knn(book_title, n=3)
    
    if similar_books.empty:
        return jsonify({
            'fulfillmentText': f'Sorry, no similar books found for "{book_title}". Would you like to try a different title?',
            'outputContexts': [{
                'name': build_context_name('has_book_context', req),
                'lifespanCount': 5,
                'parameters': {
                    'book_title': book_title
                }
            }]
        })
    
    response_text = f"Here are some books similar to \"{book_title}\":\n\n"
    for _, book in similar_books.iterrows():
        response_text += f"• {book['book_title']} – Author: {book['book_authors']}, Rating: {book['book_rating']:.1f}\n"
    response_text += "\nWould you like more recommendations?"
    
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
