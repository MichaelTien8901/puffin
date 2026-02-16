"""Web scraper for financial data from various sources."""

import time
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


class WebScraper:
    """Web scraper for financial data with rate limiting and retry logic.

    Examples:
        >>> scraper = WebScraper()
        >>> transcripts = scraper.scrape_earnings_transcripts("AAPL")
        >>> filings = scraper.scrape_sec_filings("0000320193")
    """

    def __init__(
        self,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        timeout: int = 10,
    ):
        """Initialize the web scraper.

        Args:
            rate_limit: Minimum seconds between requests
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_request_time = 0

        # User agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        ]
        self.current_ua_index = 0

    def _get_headers(self) -> dict:
        """Get request headers with rotated user agent."""
        headers = {
            "User-Agent": self.user_agents[self.current_ua_index],
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return headers

    def _rate_limit_wait(self):
        """Wait to respect rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _fetch_with_retry(self, url: str) -> Optional[requests.Response]:
        """Fetch URL with retry logic.

        Args:
            url: URL to fetch

        Returns:
            Response object or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()
                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    print(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None

    def scrape_earnings_transcripts(self, ticker: str) -> pd.DataFrame:
        """Scrape earnings call transcripts for a given ticker.

        This is a template implementation. In production, you would use:
        - SeekingAlpha API (requires subscription)
        - MotleyFool transcripts
        - Company investor relations pages

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with columns: date, quarter, transcript_text, url
        """
        # Template implementation - would need specific website implementation
        results = []

        # Example: Mock structure for demonstration
        # In production, implement actual scraping logic for specific sources
        url = f"https://example.com/earnings/{ticker}"
        response = self._fetch_with_retry(url)

        if response is None:
            return pd.DataFrame(columns=["date", "quarter", "transcript_text", "url"])

        soup = BeautifulSoup(response.content, "html.parser")

        # Example parsing logic (would be customized per source)
        # transcripts = soup.find_all("div", class_="transcript")
        # for transcript in transcripts:
        #     results.append({
        #         "date": transcript.find("span", class_="date").text,
        #         "quarter": transcript.find("span", class_="quarter").text,
        #         "transcript_text": transcript.find("div", class_="content").text,
        #         "url": transcript.find("a")["href"]
        #     })

        return pd.DataFrame(results)

    def scrape_financial_tables(self, url: str) -> pd.DataFrame:
        """Scrape financial tables from a webpage.

        Args:
            url: URL containing financial tables

        Returns:
            DataFrame with scraped table data
        """
        response = self._fetch_with_retry(url)

        if response is None:
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, "html.parser")

        # Find all tables
        tables = soup.find_all("table")

        if not tables:
            return pd.DataFrame()

        # Try to parse tables
        all_data = []
        for table in tables:
            try:
                # Extract headers
                headers = []
                header_row = table.find("thead") or table.find("tr")
                if header_row:
                    headers = [th.text.strip() for th in header_row.find_all(["th", "td"])]

                # Extract rows
                rows = []
                tbody = table.find("tbody") or table
                for tr in tbody.find_all("tr"):
                    cells = [td.text.strip() for td in tr.find_all(["td", "th"])]
                    if cells and len(cells) == len(headers):
                        rows.append(cells)

                if headers and rows:
                    df = pd.DataFrame(rows, columns=headers)
                    all_data.append(df)
            except Exception as e:
                print(f"Error parsing table: {e}")
                continue

        # Combine all tables if multiple found
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def scrape_sec_filings(self, cik: str, form_type: str = "10-K") -> pd.DataFrame:
        """Scrape SEC EDGAR filings for a company.

        Args:
            cik: Central Index Key (SEC company identifier)
            form_type: Type of form (10-K, 10-Q, 8-K, etc.)

        Returns:
            DataFrame with columns: filing_date, form_type, url, accession_number
        """
        # Pad CIK with leading zeros to 10 digits
        cik = cik.zfill(10)

        # SEC EDGAR search URL
        base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": form_type,
            "dateb": "",
            "owner": "exclude",
            "count": "100",
        }

        # Build URL with parameters
        url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

        response = self._fetch_with_retry(url)

        if response is None:
            return pd.DataFrame(columns=["filing_date", "form_type", "url", "accession_number"])

        soup = BeautifulSoup(response.content, "html.parser")

        results = []

        # Find the table containing filings
        filing_table = soup.find("table", class_="tableFile2")

        if filing_table:
            rows = filing_table.find_all("tr")[1:]  # Skip header row

            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 4:
                    form = cells[0].text.strip()
                    filing_date = cells[3].text.strip()

                    # Get document link
                    link = cells[1].find("a")
                    if link:
                        doc_url = urljoin("https://www.sec.gov", link["href"])

                        # Extract accession number from URL
                        accession = ""
                        if "Accession-Number=" in doc_url:
                            accession = doc_url.split("Accession-Number=")[1].split("&")[0]

                        results.append({
                            "filing_date": filing_date,
                            "form_type": form,
                            "url": doc_url,
                            "accession_number": accession,
                        })

        return pd.DataFrame(results)
