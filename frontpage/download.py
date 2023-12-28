import datetime as dt
import logging
from pathlib import Path
import srsly
import tqdm
import arxiv
from retry import retry
import spacy

# Assuming the ArxivArticle is defined with properties
# `created`, `title`, `abstract`, `sentences`, and `url`.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_article_age_in_days(article_published_date):
    """Calculate the age of an article in days."""
    now = dt.datetime.now(dt.timezone.utc)
    return (now - article_published_date).total_seconds() / (3600 * 24)

def parse_article_result_to_dict(article_result, nlp):
    """Parse arxiv.Result to dictionary representing ArxivArticle."""
    summary = article_result.summary.replace("\n", " ")
    doc = nlp(summary)
    sentences = [s.text for s in doc.sents]
    
    return {
        "created": str(article_result.published)[:19],
        "title": str(article_result.title),
        "abstract": summary,
        "sentences": sentences,
        "url": str(article_result.entry_id)
    }

@retry(tries=5, delay=1, backoff=2)
def fetch_and_filter_articles(nlp, query="cs", max_results=200, days_limit=2.5):
    """Fetch and filter articles from Arxiv."""
    search_results = arxiv.Search(
        query="and",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    
    results = [result for result in search_results.results() 
               if get_article_age_in_days(result.published) < days_limit 
               and result.primary_category.startswith(query)]
    logger.info(f"Found {len(results)} new results within {days_limit} days in the {query} category.")
    return results

def save_new_articles(new_articles):
    """Save new articles in the data/downloads directory."""
    if new_articles:
        filename = f"{dt.datetime.now().replace(microsecond=0).isoformat()}-articles.jsonl"
        srsly.write_jsonl(Path("data/downloads") / filename, new_articles)
        logger.info(f"Wrote {len(new_articles)} new articles into {filename}.")

def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    
    fetched_articles = fetch_and_filter_articles(nlp=nlp) # added default parameters
    
    parsed_articles = [parse_article_result_to_dict(article, nlp=nlp) for article in tqdm.tqdm(fetched_articles)]
    
    # load previously saved articles to check for duplicates
    most_recent_file = sorted(Path("data/downloads").glob("*.jsonl"))[-1]
    old_articles_dict = {art['title']: art for art in srsly.read_jsonl(most_recent_file)}
    
    new_articles = [art for art in parsed_articles if art['title'] not in old_articles_dict]
    
    save_new_articles(new_articles)

if __name__ == "__main__":
    main()