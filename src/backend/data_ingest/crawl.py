"""To run
python -m src.backend.data_ingest.crawl
"""
import os
import asyncio
import logging
import hydra
from datetime import datetime
from omegaconf import DictConfig
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter


logger = logging.getLogger(__name__)


async def crawler_main(cfg):
    combined_filter = URLPatternFilter(patterns=[
        "*grc*",                         # Match GRC pages
        "*smc*",                         # Match SMC pages
        "*/category/featured/page/*"     # Match pagination pages
    ])

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            filter_chain=FilterChain([combined_filter]),
            include_external=False,
            max_pages=100,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True
    )

    results = []
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            cfg.crawler.url,
            config=config
        )
        for result in results:
            depth = result.metadata.get("depth", 0)
            print(f"Depth: {depth} | {result.url}")
    
    # Create output directory
    output_dir = cfg.crawler.raw_md_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for the file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{cfg.crawler.file_name}_{timestamp}.md"
    output_file_path = os.path.join(output_dir, output_file)
    
    # Also create a symlink to the latest file for easy access
    latest_link = os.path.join(output_dir, f"{cfg.crawler.file_name}_latest.md")
    
    # Remove the old symlink if it exists
    if os.path.exists(latest_link):
        os.remove(latest_link)
        
    # Create a new symlink pointing to the latest file
    try:
        os.symlink(output_file_path, latest_link)
        logger.info(f"Created symlink to latest file: {latest_link}")
    except Exception as e:
        # On Windows, symlinks might require admin privileges
        logger.warning(f"Could not create symlink: {e}")
        # Copy the file instead as a fallback
        import shutil
        shutil.copy2(output_file_path, latest_link)
        logger.info(f"Created copy of latest file: {latest_link}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        # Add timestamp to the file content
        crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"# Crawl Results - {crawl_time}\n\n")
        f.write(f"Total pages crawled: {len(results)}\n\n")
        
        # Group by depth
        depth_counts = {}
        for result in results:
            depth = result.metadata.get("depth", 0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        f.write("## Pages crawled by depth:\n\n")
        for depth, count in sorted(depth_counts.items()):
            f.write(f"- Depth {depth}: {count} pages\n")
        f.write("\n")
        
        # Write detailed results for each page
        f.write("## Detailed Results\n\n")
        for result in results:
            title = result.metadata.get("title", "Untitled")
            depth = result.metadata.get("depth", 0)
            
            f.write(f"### {title}\n\n")
            f.write(f"- URL: {result.url}\n")
            f.write(f"- Depth: {depth}\n\n")
            
            # Add content summary
            if hasattr(result, 'markdown') and result.markdown:
                f.write("**Content:**\n\n")
                f.write(result.markdown + "\n\n")
            
            f.write("---\n\n")


@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="data_ingest")
def main(cfg) -> None:
    asyncio.run(crawler_main(cfg))


if __name__ == "__main__":
    main()