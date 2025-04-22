import logging
import os
import json
import re
from typing import List
from datetime import datetime
from dotenv import load_dotenv
from omegaconf import DictConfig
from pydantic_ai import Agent
from src.backend.data_ingest.extract_crawl_model import (
    ExtractedElectionData,
    ExtractedConstituency
)


load_dotenv()

logger = logging.getLogger(__name__)


class ElectionDataExtractor:
    def __init__(self, cfg: DictConfig, use_existing_data=True):
        self.cfg = cfg
        self.model = cfg.crawler.llm
        self.prompts = cfg.extract_crawl_prompts
        self.use_existing_data = use_existing_data
        self.extracted_dir = cfg.crawler.extracted_dir
        self.existing_data = self._load_existing_data() if use_existing_data else {}
        logger.info(f"Initialized ElectionDataExtractor with model: {self.model}")
        
    def _load_existing_data(self):
        """Load existing constituency data from JSON files."""
        existing_data = {}
        latest_file = os.path.join(self.extracted_dir, f"{self.cfg.crawler.file_name}_latest.json")
        
        if os.path.exists(latest_file):
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Create a dictionary of constituency data for quick lookup
                    if 'constituencies' in data:
                        for constituency in data['constituencies']:
                            if 'name' in constituency:
                                existing_data[constituency['name'].lower().strip()] = constituency
                logger.info(f"Loaded {len(existing_data)} constituencies from existing data")
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")
        
        return existing_data
    
    async def identify_constituencies(self, content: str) -> List[str]:
        """Identify list of all constituencies in the content."""
        constituency_agent = Agent(
            model=self.model,
            result_type=List[str],
            system_prompt=self.prompts.identify_constituencies_prompt
        )
        result = await constituency_agent.run(content)
        logger.info(f"Identified constituencies: {result.data}")
        constituencies_names = result.data
        return constituencies_names
    
    async def extract_constituency_details(
        self, content: str, constituency_name: str
    ) -> ExtractedConstituency:
        """Extract detailed information about a specific constituency.
        
        Uses existing data if available to avoid re-extracting the same constituency data.
        """
        # Check if we have this constituency in existing data
        cache_key = constituency_name.lower().strip()
        
        if self.use_existing_data and cache_key in self.existing_data:
            cached_data = self.existing_data[cache_key]
            logger.info(f"Using existing data for {constituency_name}")
            
            # Convert the cached dictionary back to a Pydantic model
            return ExtractedConstituency(**cached_data)
        
        # Not in existing data, extract using LLM
        logger.info(f"Extracting new data for {constituency_name}")
        details_agent = Agent(
            model=self.model,
            result_type=ExtractedConstituency,
            system_prompt=self.prompts.extract_constituency_details_prompt.format(
                constituency_name=constituency_name
            ),
            max_retries=3
        )
        
        try:
            result = await details_agent.run(content)
            logger.info(f"Successfully extracted details for {constituency_name}")
            return result.data
        except Exception as e:
            logger.error(f"Error extracting details for {constituency_name}: {str(e)}")
            # Return minimal valid data as fallback
            return ExtractedConstituency(
                name=constituency_name,
                constituency_type="Unknown",
                contesting_parties=[],
                candidates=[]
            )
    
    async def extract_election_data(self, content: str) -> ExtractedElectionData:
        """Main method to extract election data using a two-step process."""
        # Step 1: Identify all constituencies
        constituency_names = await self.identify_constituencies(content)
        logger.info(f"Found {len(constituency_names)} "
                f"constituencies: {', '.join(constituency_names)}")
        
        # Step 2: Extract details for each constituency
        constituency_details = []
        for name in constituency_names:
            try:
                details = await self.extract_constituency_details(content, name)
                constituency_details.append(details)
                logger.info(f"Extracted details for {name}")
            except Exception as e:
                logger.error(f"Error extracting details for {name}: {e}")
        return ExtractedElectionData(constituencies=constituency_details)


def preprocess_markdown(content: str) -> str:
    """Clean up markdown content by removing unnecessary elements.
    
    Args:
        content: Raw markdown content
        
    Returns:
        Cleaned markdown content with unnecessary elements removed
    """
    # Remove markdown image links: ![alt text](image_url)
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    
    # Remove markdown links but keep the text: [text](url) -> text
    content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)
    
    # Remove HTML image tags: <img src="..." />
    content = re.sub(r'<img.*?/?>', '', content)
    
    # Remove HTML anchor tags but keep the text: <a href="...">text</a> -> text
    content = re.sub(r'<a.*?>(.*?)</a>', r'\1', content)
    
    # Remove Love0 and similar reactions
    content = re.sub(r'\[ Love\d+\].*?".*?"', '', content)
    
    # Remove read time indicators
    content = re.sub(r'\d+ min read', '', content)
    content = re.sub(r'< \d+ min read', '', content)
    
    # Remove date indicators with bullet points
    content = re.sub(r'\d+ [A-Za-z]+ \d{4} •', '', content)
    
    # Remove multiple consecutive newlines (replace with double newline)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove category tags like [#GE2025][Featured][News]
    content = re.sub(r'\[#[^\]]+\]\[[^\]]+\]\[[^\]]+\]', '', content)
    
    return content

def read_markdown_file(file_path: str) -> str:
    """Read a markdown file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""



async def llm_extract_main(cfg: DictConfig) -> None:
    """Main entry point for the election data extraction process."""
    # Get extraction settings from config
    use_existing_data = cfg.crawler.use_existing_json
    input_file = os.path.join(
        cfg.crawler.raw_md_dir,
        f"{cfg.crawler.file_name}_latest.md"
    )
    base_name = cfg.crawler.file_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = cfg.crawler.extracted_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{base_name}_{timestamp}.json"
    latest_output = f"{output_dir}/{base_name}_latest.json"

    logger.info(f"Reading content from {input_file}")
    content = read_markdown_file(input_file)
    if not content:
        logger.error("No content to process")
        return
    
    # Save cleaned markdown
    logger.info("Preprocessing markdown content to remove unnecessary elements")
    cleaned_content = preprocess_markdown(content)
    logger.info(f"Original content length: {len(content)}, Cleaned content length: {len(cleaned_content)}")
    cleaned_file = f"{output_dir}/{base_name}_cleaned_{timestamp}.md"
    with open(cleaned_file, "w", encoding="utf-8") as f:
        f.write(cleaned_content)
    logger.info(f"Saved cleaned content to {cleaned_file}")
    
    # Extact and save in JSON format
    content = cleaned_content
    extractor = ElectionDataExtractor(cfg, use_existing_data)
    logger.info(f"Starting extraction process... (using existing data: {use_existing_data})")
    if not use_existing_data:
        logger.info("Will re-extract all constituencies, even if they exist in previous data")
    extracted_data = await extractor.extract_election_data(content)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data.model_dump(), f, indent=2)
    
    # Create/update the latest symlink or copy
    if os.path.exists(latest_output):
        os.remove(latest_output)
    
    try:
        os.symlink(output_file, latest_output)
        logger.info(f"Created symlink to latest output: {latest_output}")
    except Exception as e:
        # Fallback for systems where symlinks require privileges
        logger.warning(f"Could not create symlink: {e}")
        import shutil
        shutil.copy2(output_file, latest_output)
        logger.info(f"Created copy of latest output: {latest_output}")
    
    logger.info(f"Extracted {len(extracted_data.constituencies)} constituencies")
    logger.info(f"Results saved to {output_file}")
    
    # Return the extracted data so it can be used by other components
    return extracted_data


