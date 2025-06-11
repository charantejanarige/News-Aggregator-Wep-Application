from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import torch
import re
import os
import requests
import numpy as np
from bs4 import BeautifulSoup
from torch import nn
from torch.serialization import add_safe_globals
from sklearn.preprocessing import LabelEncoder
import validators
import json
from dotenv import load_dotenv
import random
from datetime import datetime, timedelta
import traceback
import time
import logging
import urllib.parse
import google.generativeai as genai
from google.genai import types
import sqlite3
import hashlib
import uuid
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Add LabelEncoder to safe globals to allow loading it with weights_only=True
add_safe_globals([LabelEncoder])

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Google Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Define the hybrid model (CNN + RNN) - same architecture as in training
class HybridTextModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes, hidden_size, num_layers, num_classes, dropout=0.5):
        super(HybridTextModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # CNN part
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_size)) for k in filter_sizes
        ])
        
        # RNN part (LSTM)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout and fully connected layers
        self.dropout = nn.Dropout(dropout)
        
        # Combine CNN and RNN features
        cnn_output_size = num_filters * len(filter_sizes)
        lstm_output_size = hidden_size * 2  # bidirectional
        
        self.fc1 = nn.Linear(cnn_output_size + lstm_output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, max_len)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, max_len, embed_size)
        
        # CNN part
        cnn_input = embedded.unsqueeze(1)  # (batch_size, 1, max_len, embed_size)
        
        # Apply convolutions and max pooling
        pooled_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(cnn_input))  # (batch_size, num_filters, max_len-filter_size+1, 1)
            pooled = torch.max_pool2d(conv_out, 
                                     (conv_out.size(2), 1))  # (batch_size, num_filters, 1, 1)
            pooled = pooled.squeeze(3).squeeze(2)  # (batch_size, num_filters)
            pooled_outputs.append(pooled)
        
        # Concatenate pooled outputs
        cnn_features = torch.cat(pooled_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # RNN part
        lstm_out, _ = self.lstm(embedded)  # (batch_size, max_len, hidden_size*2)
        
        # Get the last time step output
        lstm_features = lstm_out[:, -1, :]  # (batch_size, hidden_size*2)
        
        # Combine CNN and RNN features
        combined_features = torch.cat([cnn_features, lstm_features], dim=1)
        
        # Apply dropout and classification layers
        combined_features = self.dropout(combined_features)
        hidden = self.relu(self.fc1(combined_features))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        
        return logits

# Text preprocessing function - same as in training
def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Simple tokenization by space
    tokens = text.split()
    # Common stopwords (small set)
    stop_words = {'the', 'a', 'an', 'and', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Convert text to sequence - same as in training
def text_to_sequence(text, word2idx, max_length=50):
    words = text.split()
    seq = [word2idx.get(word, word2idx['<UNK>']) for word in words[:max_length]]

    # Pad sequence
    if len(seq) < max_length:
        seq = seq + [word2idx['<PAD>']] * (max_length - len(seq))

    return seq

# Function to make predictions on new text
def predict_category(text, model, word2idx, label_encoder, device, max_length=50):
    # Preprocess text
    processed_text = preprocess_text(text)

    # Convert to sequence
    seq = text_to_sequence(processed_text, word2idx, max_length)

    # Convert to tensor
    seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(seq_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    # Get category
    category_id = predicted.item()
    category = label_encoder.inverse_transform([category_id])[0]

    return category, float(confidence.item())

# Function to validate image URL
def is_valid_image_url(url):
    if not url or not isinstance(url, str):
        return False

    if not url.startswith(('http://', 'https://')):
        return False

    # Check if URL ends with common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
    if any(url.lower().endswith(ext) for ext in image_extensions):
        return True

    # If not ending with image extension, try to check the content type
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/*'
        }
        response = requests.head(url, headers=headers, timeout=3, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '')
        return content_type.startswith('image/')
    except Exception as e:
        logger.debug(f"Error validating image URL {url}: {e}")
        # If we can't check the content type, assume it's not an image
        return False

# Function to normalize URL
def normalize_url(url, base_url):
    if not url:
        return None
        
    # Remove query parameters and fragments
    url = url.split('#')[0]

    # Make absolute URL
    if not url.startswith(('http://', 'https://')):
        if url.startswith('//'):
            return 'https:' + url
        elif url.startswith('/'):
            return base_url + url
        else:
            return base_url + '/' + url

    return url

# Function to extract article details from a news website
def extract_article_details(article_url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(article_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        base_url = get_base_url(article_url)
        
        # Extract title
        title = None
        # Try meta title first
        meta_title = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'twitter:title'})
        if meta_title and meta_title.get('content'):
            title = meta_title.get('content')
        # If no meta title, try h1
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.text.strip()
        # If still no title, use the page title
        if not title:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.text.strip()
        
        # Extract description
        description = None
        meta_desc = soup.find('meta', property='og:description') or soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc.get('content')
        
        # Extract image
        image_url = None
        
        # Try to find meta og:image
        meta_og_image = soup.find('meta', property='og:image')
        if meta_og_image and meta_og_image.get('content'):
            img_url = meta_og_image.get('content')
            img_url = normalize_url(img_url, base_url)
            if is_valid_image_url(img_url):
                image_url = img_url
                logger.info(f"Found og:image: {image_url}")
            
        # Try to find meta twitter:image
        if not image_url:
            meta_twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if meta_twitter_image and meta_twitter_image.get('content'):
                img_url = meta_twitter_image.get('content')
                img_url = normalize_url(img_url, base_url)
                if is_valid_image_url(img_url):
                    image_url = img_url
                    logger.info(f"Found twitter:image: {image_url}")
        
        # Try to find article:image
        if not image_url:
            meta_article_image = soup.find('meta', property='article:image')
            if meta_article_image and meta_article_image.get('content'):
                img_url = meta_article_image.get('content')
                img_url = normalize_url(img_url, base_url)
                if is_valid_image_url(img_url):
                    image_url = img_url
                    logger.info(f"Found article:image: {image_url}")
        
        # Try to find the main article container
        article_container = None
        for selector in ['article', '[role="article"]', '.article', '#article', '.post', '.story', '.news-item']:
            container = soup.select_one(selector)
            if container:
                article_container = container
                break
        
        # If no specific article container found, use the body
        if not article_container:
            article_container = soup.body
        
        # Try to find the first large image in the article container
        if not image_url and article_container:
            # Look for figure with image first
            figures = article_container.find_all('figure')
            for figure in figures:
                img = figure.find('img')
                if img:
                    # Skip small icons and logos
                    width = img.get('width')
                    height = img.get('height')
                    if width and height and (int(width) < 100 or int(height) < 100):
                        continue
                        
                    # Check for common classes that indicate main images
                    img_class = img.get('class', [])
                    if img_class:
                        img_class_str = ' '.join(img_class).lower()
                        if any(term in img_class_str for term in ['thumb', 'icon', 'logo', 'avatar']):
                            continue
                            
                    # Get image source
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        src = normalize_url(src, base_url)
                        if is_valid_image_url(src):
                            image_url = src
                            logger.info(f"Found image in figure: {image_url}")
                            break
            
            # If no image found in figures, look for any large image
            if not image_url:
                images = article_container.find_all('img')
                for img in images:
                    # Skip small icons and logos
                    width = img.get('width')
                    height = img.get('height')
                    if width and height and (int(width) < 100 or int(height) < 100):
                        continue
                        
                    # Check for common classes that indicate main images
                    img_class = img.get('class', [])
                    if img_class:
                        img_class_str = ' '.join(img_class).lower()
                        if any(term in img_class_str for term in ['thumb', 'icon', 'logo', 'avatar']):
                            continue
                            
                    # Get image source
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        src = normalize_url(src, base_url)
                        if is_valid_image_url(src):
                            image_url = src
                            logger.info(f"Found image in article: {image_url}")
                            break
        
        # Extract date
        date = None
        # Try meta date
        meta_date = soup.find('meta', property='article:published_time')
        if meta_date and meta_date.get('content'):
            date = meta_date.get('content')
        # Try time tag
        if not date:
            time_tag = soup.find('time')
            if time_tag:
                date_attr = time_tag.get('datetime')
                if date_attr:
                    date = date_attr
                else:
                    date = time_tag.text.strip()
        
        # Extract author
        author = None
        meta_author = soup.find('meta', property='article:author') or soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            author = meta_author.get('content')
        
        # Extract content
        content = ""
        if article_container:
            # Try to find the main content
            paragraphs = article_container.find_all('p')
            for p in paragraphs:
                # Skip if paragraph is too short or likely to be a caption
                if len(p.text.strip()) < 20:
                    continue
                
                # Skip if paragraph is in a figure or figcaption
                if p.find_parent('figure') or p.find_parent('figcaption'):
                    continue
                
                # Add paragraph text to content
                content += p.text.strip() + "\n\n"
        
        # Return article details
        return {
            'title': title,
            'description': description,
            'image_url': image_url,
            'date': date,
            'author': author,
            'url': article_url,
            'hasImage': image_url is not None,
            'content': content
        }

    except Exception as e:
        logger.error(f"Error extracting article details from {article_url}: {e}")
        return None

# Function to scrape news articles from a given URL
def scrape_articles(url=None):
    if not url:
        url = "https://timesofindia.indiatimes.com/"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        base_url = get_base_url(url)
        
        # Store found articles
        articles = []
        article_urls = set()
        
        # Method 1: Look for article elements
        article_elements = soup.find_all(['article', 'div', 'li'], class_=lambda c: c and any(x in str(c).lower() for x in ['article', 'story', 'news', 'headline', 'item']))
        
        for article in article_elements:
            # Find the link
            link = article.find('a')
            if not link or not link.get('href'):
                continue
                
            # Get the URL
            article_url = link.get('href')
            article_url = normalize_url(article_url, base_url)
            
            if not article_url or not article_url.startswith(('http://', 'https://')):
                continue
                
            # Skip if already found
            if article_url in article_urls:
                continue
                
            # Get the title
            title = None
            
            # Try to find title in heading tags
            heading = article.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if heading:
                title = heading.text.strip()
            
            # If no heading, use link text
            if not title and link.text.strip():
                title = link.text.strip()
            
            # Skip if no title or title too short
            if not title or len(title) < 15:
                continue
                
            # Find image
            image_url = None
            img = article.find('img')
            if img:
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    src = normalize_url(src, base_url)
                    if is_valid_image_url(src):
                        image_url = src
            
            # Add to articles
            article_urls.add(article_url)
            articles.append({
                'title': title,
                'link': article_url,
                'image_url': image_url,
                'hasImage': image_url is not None
            })
        
        # Method 2: Look for links with specific patterns
        if len(articles) < 5:
            # Find all links
            links = soup.find_all('a')
            
            for link in links:
                # Skip if no href
                if not link.get('href'):
                    continue
                    
                # Get the URL
                article_url = link.get('href')
                article_url = normalize_url(article_url, base_url)
                
                if not article_url or not article_url.startswith(('http://', 'https://')):
                    continue
                    
                # Skip if already found
                if article_url in article_urls:
                    continue
                    
                # Check if URL contains article indicators
                url_lower = article_url.lower()
                if not any(x in url_lower for x in ['/article/', '/story/', '/news/', '/post/']):
                    continue
                    
                # Get the title
                title = link.text.strip()
                
                # Skip if no title or title too short
                if not title or len(title) < 15:
                    continue
                    
                # Find image
                image_url = None
                img = link.find('img')
                if img:
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        src = normalize_url(src, base_url)
                        if is_valid_image_url(src):
                            image_url = src
                
                # Add to articles
                article_urls.add(article_url)
                articles.append({
                    'title': title,
                    'link': article_url,
                    'image_url': image_url,
                    'hasImage': image_url is not None
                })
        
        # Method 3: Look for common news article containers
        if len(articles) < 5:
            # Find all divs with specific classes
            containers = soup.find_all(['div', 'section'], class_=lambda c: c and any(x in str(c).lower() for x in ['content', 'main', 'container', 'wrapper', 'body']))
            
            for container in containers:
                # Find all links
                links = container.find_all('a')
                
                for link in links:
                    # Skip if no href
                    if not link.get('href'):
                        continue
                        
                    # Get the URL
                    article_url = link.get('href')
                    article_url = normalize_url(article_url, base_url)
                    
                    if not article_url or not article_url.startswith(('http://', 'https://')):
                        continue
                        
                    # Skip if already found
                    if article_url in article_urls:
                        continue
                        
                    # Get the title
                    title = link.text.strip()
                    
                    # Skip if no title or title too short
                    if not title or len(title) < 15:
                        continue
                        
                    # Find image
                    image_url = None
                    parent = link.parent
                    if parent:
                        img = parent.find('img')
                        if img:
                            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                            if src:
                                src = normalize_url(src, base_url)
                                if is_valid_image_url(src):
                                    image_url = src
                    
                    # Add to articles
                    article_urls.add(article_url)
                    articles.append({
                        'title': title,
                        'link': article_url,
                        'image_url': image_url,
                        'hasImage': image_url is not None
                    })
        
        # Fetch full details for each article
        detailed_articles = []
        for article in articles[:15]:  # Limit to 15 articles to avoid too many requests
            logger.info(f"Fetching details for article: {article['title']}")
            details = extract_article_details(article['link'])
            
            if details:
                # If we already have an image but the details found a better one, use that
                if not article['hasImage'] and details['hasImage']:
                    article['image_url'] = details['image_url']
                    article['hasImage'] = True
                
                # If details didn't find an image but we have one, keep ours
                if article['hasImage'] and not details['hasImage']:
                    details['image_url'] = article['image_url']
                    details['hasImage'] = True
                
                # Keep our title if details didn't find one
                if not details['title'] and article['title']:
                    details['title'] = article['title']
                
                detailed_articles.append(details)
            else:
                # If details extraction failed, use what we have
                detailed_articles.append(article)
        
        # If we have detailed articles, use those
        if detailed_articles:
            return detailed_articles
        
        # Otherwise, return the original articles
        return articles

    except Exception as e:
        logger.error(f"Error scraping articles: {e}")
        return []

# Helper function to get base URL
def get_base_url(url):
    parts = urllib.parse.urlparse(url)
    return f"{parts.scheme}://{parts.netloc}"

# Function to validate URL
def is_valid_url(url):
    try:
        return validators.url(url)
    except:
        # If validators package is not available, do a simple check
        return url.startswith(('http://', 'https://'))

# Load model function
def load_model():
    model_path = 'best_model.pth'

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}.")
        return None, None, None

    try:
        # Load the model
        try:
            # Method 1: Using safe globals (preferred, more secure)
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as e:
            # Method 2: Fallback to weights_only=False (less secure)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Model parameters
        embed_size = 128
        num_filters = 100
        filter_sizes = [3, 4, 5]
        hidden_size = 128
        num_layers = 2
        dropout = 0.5
        
        # Initialize model with saved parameters
        model = HybridTextModel(
            vocab_size=len(checkpoint['word2idx']),
            embed_size=embed_size,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=len(checkpoint['label_encoder'].classes_),
            dropout=dropout
        ).to(device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        word2idx = checkpoint['word2idx']
        label_encoder = checkpoint['label_encoder']
        
        return model, word2idx, label_encoder

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None

# Function to scrape trending news from popular news sites
def get_trending_news():
    # List of news sources to try
    news_sources = [
        {"url": "https://www.bbc.com/news", "name": "BBC News"},
        {"url": "https://www.ndtv.com/", "name": "NDTV"},
        {"url": "https://timesofindia.indiatimes.com/", "name": "Times of India"},
        {"url": "https://www.indiatoday.in/", "name": "India Today"},
        {"url": "https://www.hindustantimes.com/", "name": "Hindustan Times"},
        {"url": "https://www.theguardian.com/international", "name": "The Guardian"},
        {"url": "https://www.cnn.com/", "name": "CNN"}
    ]

    # Shuffle the sources to avoid always hitting the same one first
    random.shuffle(news_sources)

    all_articles = []

    # Try each source until we get enough articles
    for source in news_sources:
        if len(all_articles) >= 12:
            break
            
        try:
            logger.info(f"Trying to fetch news from {source['name']}")
            
            # Scrape articles from the source
            articles = scrape_articles(source['url'])
            
            # Add source name to each article
            for article in articles:
                article['source'] = source['name']
                
                # Add publishedAt if not present
                if not article.get('date'):
                    article['publishedAt'] = datetime.now().isoformat()
                else:
                    article['publishedAt'] = article['date']
                
                # Rename link to url if needed
                if article.get('link') and not article.get('url'):
                    article['url'] = article['link']
                
                # Rename image_url to urlToImage if needed
                if article.get('image_url') and not article.get('urlToImage'):
                    article['urlToImage'] = article['image_url']
            
            # Add to all articles
            all_articles.extend(articles)
            logger.info(f"Found {len(articles)} articles from {source['name']}")
            
        except Exception as e:
            logger.error(f"Error fetching from {source['name']}: {e}")
            traceback.print_exc()
            continue

    # If we still don't have enough articles, use fallback data
    if len(all_articles) < 5:
        logger.warning("Using fallback news data")
        all_articles = get_fallback_news()

    # Remove duplicates and limit to 12 articles
    unique_articles = []
    seen_titles = set()

    for article in all_articles:
        if article.get('title') and article['title'] not in seen_titles:
            seen_titles.add(article['title'])
            
            # Ensure the article has all required fields
            if not article.get('description'):
                article['description'] = f"Read the latest news from {article['source']}"
                
            # Final validation of image URL
            if article.get('urlToImage'):
                article['hasImage'] = is_valid_image_url(article['urlToImage'])
                if article['hasImage']:
                    logger.info(f"Valid image for article: {article['title']} - {article['urlToImage']}")
                else:
                    logger.warning(f"Invalid image for article: {article['title']} - {article['urlToImage']}")
            else:
                article['hasImage'] = False
            
            unique_articles.append(article)
            if len(unique_articles) >= 12:
                break

    # Log summary of articles with images
    articles_with_images = sum(1 for article in unique_articles if article['hasImage'])
    logger.info(f"Total articles: {len(unique_articles)}, Articles with images: {articles_with_images}")

    return unique_articles

# Fallback news data in case all scraping fails
def get_fallback_news():
    # Generate dates for the last few days
    today = datetime.now()
    dates = [(today - timedelta(days=i)).isoformat() for i in range(10)]

    return [
        {
            'title': 'Global Leaders Gather for Climate Summit to Address Environmental Challenges',
            'description': 'World leaders meet to discuss urgent climate action and set new emission reduction targets.',
            'url': 'https://example.com/climate-summit',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[0],
            'source': 'Global News'
        },
        {
            'title': 'Tech Giants Announce Breakthrough in Quantum Computing Research',
            'description': 'Major technology companies reveal significant advancements in quantum computing capabilities.',
            'url': 'https://example.com/quantum-computing',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[1],
            'source': 'Tech Today'
        },
        {  'hasImage': False,
            'publishedAt': dates[1],
            'source': 'Tech Today'
        },
        {
            'title': 'Medical Researchers Develop New Treatment for Chronic Disease',
            'description': 'Scientists announce promising results from clinical trials of innovative therapy.',
            'url': 'https://example.com/medical-breakthrough',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[2],
            'source': 'Health News'
        },
        {
            'title': 'Economic Recovery Shows Strong Signs as Global Markets Rebound',
            'description': 'Financial indicators point to sustained economic growth following recent challenges.',
            'url': 'https://example.com/economic-recovery',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[3],
            'source': 'Financial Times'
        },
        {
            'title': 'Space Exploration Mission Discovers Evidence of Water on Distant Planet',
            'description': 'Astronomers find compelling signs of water presence, raising possibilities for extraterrestrial life.',
            'url': 'https://example.com/space-discovery',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[4],
            'source': 'Science Daily'
        },
        {
            'title': 'Cultural Heritage Site Recognized with International Protection Status',
            'description': 'Historic landmark receives designation ensuring preservation for future generations.',
            'url': 'https://example.com/heritage-protection',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[5],
            'source': 'Culture News'
        },
        {
            'title': 'Sports Championship Concludes with Unexpected Victory in Final Match',
            'description': 'Underdog team overcomes challenges to win prestigious tournament in dramatic fashion.',
            'url': 'https://example.com/sports-championship',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[6],
            'source': 'Sports Network'
        },
        {
            'title': 'Education Reform Initiative Aims to Address Learning Gaps in Schools',
            'description': 'New policy framework focuses on inclusive education and technological integration.',
            'url': 'https://example.com/education-reform',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[7],
            'source': 'Education Weekly'
        },
        {
            'title': 'Renewable Energy Project Sets Record for Sustainable Power Generation',
            'description': 'Innovative solar and wind installation achieves unprecedented efficiency levels.',
            'url': 'https://example.com/renewable-energy',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[8],
            'source': 'Energy News'
        },
        {
            'title': 'Transportation Innovation Promises to Reduce Urban Congestion',
            'description': 'Smart city technology implementation shows significant improvements in traffic flow.',
            'url': 'https://example.com/transportation-innovation',
            'urlToImage': None,
            'hasImage': False,
            'publishedAt': dates[9],
            'source': 'Urban Development Today'
        }
    ]

# Define news sources
NEWS_SOURCES = [
    {"name": "The Hindu", "url": "https://www.thehindu.com/"},
    {"name": "India Today", "url": "https://www.indiatoday.in/"},
    {"name": "NDTV", "url": "https://www.ndtv.com/"},
    {"name": "Times of India", "url": "https://timesofindia.indiatimes.com/"},
    {"name": "News18", "url": "https://www.news18.com/"},
    {"name": "LiveMint", "url": "https://www.livemint.com/"},
    {"name": "Deccan Herald", "url": "https://www.deccanherald.com/"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/"},
    {"name": "BBC News", "url": "https://www.bbc.com/news"}
]

# Chat history storage for buffer memory
chat_history = {}

# Function to generate AI response for chatbot
def generate_ai_response(prompt, user_id=None):
    try:
        # Create a system prompt to ensure the AI only discusses news-related topics
        system_prompt = """You are a helpful news assistant. You can answer questions about current events, politics, and global affairs, and you can provide general summaries of news from well-known sources such as Times of India, BBC, or NDTV, even if the user does not provide a specific article. Only news articles related info, dont answer maths,or programming questions. If real-time data is not available, answer based on recent news knowledge as of 2025.
"""
        
        # Get chat history for this user
        history = []
        if user_id and user_id in chat_history:
            history = chat_history[user_id][-5:]  # Get last 5 messages for context
            
        # Add history to prompt if available
        context_prompt = prompt
        if history:
            context_prompt = "Previous conversation:\n" + "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history]) + "\n\nUser: " + prompt
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt
        )
        
        # Generate response
        response = model.generate_content(context_prompt)
        
        # Save to history if user_id is provided
        if user_id:
            if user_id not in chat_history:
                chat_history[user_id] = []
            chat_history[user_id].append({"user": prompt, "assistant": response.text})
            # Limit history size
            if len(chat_history[user_id]) > 20:
                chat_history[user_id] = chat_history[user_id][-20:]
        
        # Return the response text
        return response.text
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."

# Database setup for user authentication
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create saved_articles table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS saved_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        url TEXT NOT NULL,
        image_url TEXT,
        source TEXT NOT NULL,
        category TEXT NOT NULL,
        has_image BOOLEAN NOT NULL,
        saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        UNIQUE(user_id, url)
    )
    ''')
    
    conn.commit()
    conn.close()

# Hash password function
def hash_password(password):
    # Use SHA-256 for password hashing
    return hashlib.sha256(password.encode()).hexdigest()

# User registration function
def register_user(username, password, name, age, gender):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return False, "Username already exists"
        
        # Hash the password
        hashed_password = hash_password(password)
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users (username, password, name, age, gender) VALUES (?, ?, ?, ?, ?)",
            (username, hashed_password, name, int(age), gender)
        )
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        return False, "An error occurred during registration"

# User login function
def login_user(username, password):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Hash the password
        hashed_password = hash_password(password)
        
        # Check credentials
        cursor.execute("SELECT id, name FROM users WHERE username = ? AND password = ?", 
                      (username, hashed_password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return True, {"id": user[0], "name": user[1]}
        else:
            return False, "Invalid username or password"
    except Exception as e:
        logger.error(f"Error logging in user: {e}")
        return False, "An error occurred during login"

# Save article function
def save_article(user_id, article_data):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check if article is already saved
        cursor.execute("SELECT id FROM saved_articles WHERE user_id = ? AND url = ?", 
                      (user_id, article_data['url']))
        if cursor.fetchone():
            conn.close()
            return True, "Article already saved"
        
        # Insert article
        cursor.execute(
            """INSERT INTO saved_articles 
               (user_id, title, url, image_url, source, category, has_image) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, 
                article_data['title'], 
                article_data['url'], 
                article_data.get('image_url', None), 
                article_data['source'], 
                article_data['category'],
                article_data.get('has_image', False)
            )
        )
        conn.commit()
        conn.close()
        return True, "Article saved successfully"
    except Exception as e:
        logger.error(f"Error saving article: {e}")
        return False, "An error occurred while saving the article"

# Remove saved article function
def remove_saved_article(user_id, url):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Delete article
        cursor.execute("DELETE FROM saved_articles WHERE user_id = ? AND url = ?", 
                      (user_id, url))
        
        if cursor.rowcount > 0:
            conn.commit()
            conn.close()
            return True, "Article removed successfully"
        else:
            conn.close()
            return False, "Article not found"
    except Exception as e:
        logger.error(f"Error removing article: {e}")
        return False, "An error occurred while removing the article"

# Get saved articles function
def get_saved_articles(user_id):
    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        # Get all saved articles for user
        cursor.execute(
            """SELECT * FROM saved_articles 
               WHERE user_id = ? 
               ORDER BY saved_at DESC""", 
            (user_id,)
        )
        
        articles = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return articles
    except Exception as e:
        logger.error(f"Error getting saved articles: {e}")
        return []

# Get unique categories from saved articles
def get_saved_article_categories(user_id):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Get unique categories
        cursor.execute(
            """SELECT DISTINCT category FROM saved_articles 
               WHERE user_id = ? 
               ORDER BY category""", 
            (user_id,)
        )
        
        categories = [row[0] for row in cursor.fetchall()]
        conn.close()
        return categories
    except Exception as e:
        logger.error(f"Error getting saved article categories: {e}")
        return []

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

# Initialize database
init_db()

# Load model, word2idx, and label_encoder at startup
model, word2idx, label_encoder = load_model()

# Get categories
if label_encoder is not None:
    if isinstance(label_encoder.classes_, np.ndarray):
        categories = label_encoder.classes_.tolist()
    else:
        categories = list(label_encoder.classes_)
else:
    categories = []

# Cache for trending news to avoid frequent scraping
trending_news_cache = {
    'data': None,
    'timestamp': 0,
    'expiry': 30 * 60  # 30 minutes in seconds
}

@app.route('/')
def index():
    # Check if user is logged in
    is_logged_in = 'user_id' in session

    # Only fetch trending news if user is logged in
    trending_news = None
    if is_logged_in:
        # Get trending news with caching
        global trending_news_cache
        current_time = time.time()
        
        if (trending_news_cache['data'] is None or 
            current_time - trending_news_cache['timestamp'] > trending_news_cache['expiry']):
            logger.info("Fetching fresh trending news")
            trending_news = get_trending_news()
            trending_news_cache['data'] = trending_news
            trending_news_cache['timestamp'] = current_time
        else:
            logger.info("Using cached trending news")
            trending_news = trending_news_cache['data']

    return render_template('index.html', news_sources=NEWS_SOURCES, trending_news=trending_news, is_logged_in=is_logged_in)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None

    # If user is already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            error = "Please enter both username and password"
        else:
            success, result = login_user(username, password)
            if success:
                # Set session variables
                session['user_id'] = result['id']
                session['username'] = username
                session['name'] = result['name']
                
                # Redirect to home page
                return redirect(url_for('index'))
            else:
                error = result

    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None

    # If user is already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        
        # Validate form data
        if not username or not password or not name or not age or not gender:
            error = "All fields are required"
        else:
            try:
                # Convert age to integer
                age = int(age)
                if age < 13:
                    error = "You must be at least 13 years old to register"
                else:
                    # Register the user
                    success, message = register_user(username, password, name, age, gender)
                    if success:
                        # Redirect to login page
                        flash("Registration successful! Please login with your credentials.")
                        return redirect(url_for('login'))
                    else:
                        error = message
            except ValueError:
                error = "Age must be a valid number"

    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    # Clear session
    session.clear()
    return redirect(url_for('index'))

@app.route('/trending')
@login_required
def trending():
    # Get trending news with caching
    global trending_news_cache
    current_time = time.time()

    if (trending_news_cache['data'] is None or 
        current_time - trending_news_cache['timestamp'] > trending_news_cache['expiry']):
        logger.info("Fetching fresh trending news for API")
        trending_news = get_trending_news()
        trending_news_cache['data'] = trending_news
        trending_news_cache['timestamp'] = current_time
    else:
        logger.info("Using cached trending news for API")
        trending_news = trending_news_cache['data']

    return jsonify(trending_news)

@app.route('/source/<int:source_id>')
@login_required
def source_page(source_id):
    if source_id < 0 or source_id >= len(NEWS_SOURCES):
        return redirect(url_for('index'))

    source = NEWS_SOURCES[source_id]
    return render_template('source.html', source=source, categories=categories, source_id=source_id)

@app.route('/scrape/<int:source_id>', methods=['GET'])
@login_required
def scrape_source(source_id):
    if source_id < 0 or source_id >= len(NEWS_SOURCES):
        return jsonify({'error': 'Invalid source ID'}), 400

    source = NEWS_SOURCES[source_id]
    url = source['url']

    # Scrape articles
    articles = scrape_articles(url)

    if not articles:
        return jsonify({'error': 'No articles found'}), 400

    # Classify articles
    classified_articles = {}
    for category in categories:
        classified_articles[category] = []

    for article in articles:
        category, confidence = predict_category(
            article['title'], 
            model, 
            word2idx, 
            label_encoder, 
            device
        )
        
        article['confidence'] = confidence
        classified_articles[category].append(article)

    # Create summary
    summary = {category: len(classified_articles[category]) for category in categories}

    return jsonify({
        'articles': classified_articles,
        'summary': summary
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    # Get user ID from session if available
    user_id = session.get('user_id')

    # Generate AI response with chat history
    response = generate_ai_response(user_message, user_id)

    return jsonify({
        'response': response
    })

@app.route('/clear_cache')
def clear_cache():
    global trending_news_cache
    trending_news_cache['data'] = None
    trending_news_cache['timestamp'] = 0
    return redirect(url_for('index'))

@app.route('/save_article', methods=['POST'])
@login_required
def save_article_route():
    data = request.get_json()
    user_id = session.get('user_id')
    
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
    
    # Validate required fields
    required_fields = ['title', 'url', 'source', 'category']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'message': f'Missing required field: {field}'}), 400
    
    # Save article
    success, message = save_article(user_id, data)
    
    return jsonify({'success': success, 'message': message})

@app.route('/remove_saved_article', methods=['POST'])
@login_required
def remove_saved_article_route():
    data = request.get_json()
    user_id = session.get('user_id')
    
    if not data or 'url' not in data:
        return jsonify({'success': False, 'message': 'No URL provided'}), 400
    
    # Remove article
    success, message = remove_saved_article(user_id, data['url'])
    
    return jsonify({'success': success, 'message': message})

@app.route('/get_saved_articles')
@login_required
def get_saved_articles_route():
    user_id = session.get('user_id')
    
    # Get saved articles
    articles = get_saved_articles(user_id)
    
    return jsonify(articles)

@app.route('/saved_articles')
@login_required
def saved_articles_page():
    user_id = session.get('user_id')
    
    # Get saved articles
    saved_articles = get_saved_articles(user_id)
    
    # Get unique categories
    categories = get_saved_article_categories(user_id)
    
    return render_template('saved_articles.html', saved_articles=saved_articles, categories=categories)

if __name__ == '__main__':
    app.run(debug=True)
