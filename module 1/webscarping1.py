from bs4 import BeautifulSoup
import requests
import re


urls = [
    "https://www.infosys.com/",
    "https://www.infosys.com/investors.html",
    "https://www.infosys.com/about/overview.html",
    "https://www.infosys.com/about/alliances.html",
    "https://www.infosys.com/about/awards.html",
    "https://www.infosys.com/contact.html",
    "https://www.infosys.com/services/generative-ai/overview.html",
    "https://www.infosys.com/about/history.html",
    "https://www.infosys.org/infosys-foundation/initiatives.html",
    "https://www.infosys.com/leadership-institute.html",
    "https://www.infosys.com/newsroom/journalist-resources/contact.html",
    "https://www.infosys.com/about/springboard.html",
    "https://www.infosys.com/about/subsidiaries.html",
    "https://www.infosys.com/industries.html",
    "https://www.infosys.com/industries/aerospace-defense.html",
    "https://www.infosys.com/industries/agriculture.html",
    "https://www.infosys.com/industries/automotive.html",
    "https://www.infosys.com/industries/chemical-manufacturing.html",
    "https://www.infosys.com/about/management-profiles.html",
    "https://www.infosys.com/about/management-profiles/salil-parekh.html",
    "https://www.infosys.com/about/management-profiles/nandan-nilekani.html",
    "https://www.infosys.com/about/management-profiles/d-sundaram.html",
    "https://www.infosys.com/newsroom.html",
    "https://www.infosys.com/newsroom/features/2023/recognized-global-top-employer.html",
    "https://www.infosys.com/newsroom/press-releases/2024/top-3-it-services-brand-globally2024.html",
    "https://www.infosys.com/products-and-platforms.html",
    "https://www.edgeverve.com/",
    "https://www.infosysequinox.com/",
    "https://www.edgeverve.com/finacle/",
    "https://www.infosys.com/services.html",
    "https://www.infosys.com/services/engineering-services.html",
    "https://www.infosys.com/services/metaverse.html",
    "https://www.infosys.com/services/sustainability-services.html"
]

# Open the destination file in append mode
with open('./data.txt', 'a', encoding='utf-8') as file:
    # Loop over the URLs
    for url in urls:
        # Send a GET request to the URL
        page = requests.get(url)

        # Parse the HTML content
        soup = BeautifulSoup(page.text, 'html.parser')

        # Extract the text content from the parsed HTML
        text = soup.get_text(strip=True)

        # # Remove tabs, newlines, and multiple spaces
        cleaned_data = re.sub(r'[\t\n\r]+', '\n', text)
        # cleaned_data = re.sub(r'\s+', ' ', text)
        # cleaned_data = re.sub(r'\s+', '\n', cleaned_data)
        # Write the text content to the file
        file.write(cleaned_data)
        file.write('\n\n\n\n')  # Add a separator between data from different URLs