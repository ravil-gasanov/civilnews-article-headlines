import scrapy
import datetime

class CivilCrawler(scrapy.Spider):
    name = 'CivilSpider'

    start_urls = ['https://civil.ge/archives/category/news']

    def parse_contents(self, response):
        yield {
            'date-time': response.xpath('//*[@id="single-post-meta"]/span/text()').get(),
            'headline': response.xpath('.//h1/text()').get(),
            'views': response.xpath('//*[@id="single-post-meta"]/div/span[1]/text()').get(),
            'tags': response.xpath('//*[@id="the-post"]/div[2]/div/span/a/text()').getall()
        }
        

    def parse(self, response):
        post_details = response.css('div.post-details')

        i = 1
        for post in post_details:
            article_link = post.xpath('//*[@id="posts-container"]/li[' + str(i) + ']/div/a').css('a::attr("href")').get()
            
            yield response.follow(article_link, self.parse_contents)

            i += 1
        
        next_page = response.css('li.the-next-page a::attr("href")').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)