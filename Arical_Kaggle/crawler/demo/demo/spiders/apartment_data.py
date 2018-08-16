# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from demo.items import DemoItem

thefile = open('/Users/san/Dev/Arical_Kaggle/crawler/demo/demo/test.txt', 'w')



class ApartmentDataSpider(scrapy.Spider):
    name = 'apartment_data'

    def start_requests(self):
        urls = []
        for j in range(1,200):
            urls.append('http://www.century21-hk.com/eng/tran_prop.php?page={}&startwith=&isSelect=1&dcode=KHH&prop=R&bname=&searchType=B&sizeType=&year=2018&month='.format(j))

        for url in urls:
            print("Loop through URLs list")
            print(url)
            yield scrapy.Request(url=url, callback=self.parse)

    # Ver 3
    def parse(self, response):
        print("Parse Function Called")
        '''
        [QUESTION]
        Is it possible to crawl table that starts with <tf bgcolor=#000>
        '''
        num_Of_listing = 20 # Set it to 20 for testing
        for i in range(2,num_Of_listing+1):
            '''
            XPATH for Each Listing
            //*[@id="tran"]/form[2]/table/tbody/tr[2]
            //*[@id="tran"]/form[2]/table/tbody/tr[3]
            //*[@id="tran"]/form[2]/table/tbody/tr[4]
            ...
            '''
            print("i: ", i)
            curr_path = '//*[@id="tran"]/form[2]/table/tbody/tr[{}]'.format(i)

            print('curr_path: {0}'.format(curr_path))

            # For each row, it is a table
            for row in response.xpath(curr_path):
                print("ROW-Response ForLoop")
                #print(row)

                #print("Generate DemoItem Listing with XPATH")
                #listing = DemoItem()
                #listing['Input_Date'] = row1.xpath('td[2]//text()').extract_first()
                #print("DebugMessage>>listing['Input_Date']: {}".format(listing['Input_Date']))
                #listing['Address'] = row1.xpath('td[3]//text()').extract_first()
                #listing['Floor'] = row1.xpath('td[4]//text()').extract_first()
                #listing['Unit'] = row1.xpath('td[5]//text()').extract_first()
                #listing['Saleable_Area'] = row1.xpath('td[6]//text()').extract_first()
                #listing['Price_M'] = row1.xpath('td[7]//text()').extract_first()
                #listing['Price_ft2'] = row1.xpath('td[8]//text()').extract_first()
                #listing['OP_Date'] = OP_Date
                #listing['Gross_Area'] = Gross_Area
                #listing['Exp_Year'] = Exp_Year
                #listing['Facing'] = Facing
                #listing['Layout'] = Layout

                #print("Call Parse Details Function")
                print("Hard Code the deatil full url ....")
                detail_full_url = 'http://www.century21-hk.com/eng/tran_prop_detail.php?ref=06927351&year=2018'


                '''
                #[0].strip()
                #//*[@id="tran"]/form[2]/table/tbody/tr[2]/td[9]/a
                print("trial_URL testing")
                for sel in response.xpath('td[9]/a'):
                    trial_URL = sel.xpath('@href').extract()
                    print('trial_URL: {}'.format(trial_URL))

                #print('trial_URL: {}'.format(trial_URL))
                #OP_Date, Gross_Area, Exp_Year, Facing, Layout = parse_detail_page(detail_full_url)
                '''

                #OP_Date, Gross_Area, Exp_Year, Facing, Layout = parse_detail_page(detail_full_url)

                print("Generate listing as a list only")
                listing = {
                'Input_Date': row.xpath('td[2]//text()').extract_first(),
                'Address': row.xpath('td[3]//text()').extract_first(),
                'Floor': row.xpath('td[4]//text()').extract_first(),
                'Unit': row.xpath('td[5]//text()').extract_first(),
                'Saleable_Area': row.xpath('td[6]//text()').extract_first(),
                'Price_M': row.xpath('td[7]//text()').extract_first(),
                'Price_ft2': row.xpath('td[8]//text()').extract_first(),
                #'link': row.xpath('td[9]//a/@href').extract(),
                #'detail_url': 'http://www.century21-hk.com/eng/' +  str(row.xpath('td[9]//a/@href').extract()[0]),
                #'OP_Date': OP_Date,
                #'Gross_Area': Gross_Area,
                #'Exp_Year': Exp_Year,
                #'Facing': Facing,
                #'Layout': Layout
                }

                #print('DebugMessage: detail_url:')
                #url = listing['detail_url']
                #print(url)

                print("Create DemoItem Object")
                item = DemoItem()
                item['Input_Date'] = str(listing['Input_Date'])
                item['Address'] = str(listing['Address'])
                print("Test DemoItem Objet")
                print(item['Input_Date'])
                print(item['Address'])





                #print("Extract info from Detail link")
                #detail_url = str(row1.xpath('td[9]/a//@href').extract())
                #print("detail_url: {}".format(detail_url))
                #print("detail_url[0]: {}".format(detail_url[1]))
                #detail_full_url = 'http://www.century21-hk.com/eng/' + str(detail_url[1])
                #print("URL: ", detail_full_url)
                #OP_Date, Gross_Area, Exp_Year, Facing, Layout = parse_detail_page(detail_full_url)



                #print('End of adding Extrat_Text into DemoItem')
                print('Append current listing into listings array')
                thefile.write("%s\n" % listing)



    def parse_detail_page(detail_url):
        print("Parse Details Function Called")
        response = fetch(detail_url) # Get the response

        # Extract data from the Detail page
        OP_Date = response.xpath('//*[@id="trandetail"]/table/tbody/tr/td/table/tbody/tr[4]/td[4]//text()').extract_first()
        Gross_Area = response.xpath('//*[@id="trandetail"]/table/tbody/tr/td/table/tbody/tr[5]/td[2]//text()').extract_first()
        Exp_Year = response.xpath('//*[@id="trandetail"]/table/tbody/tr/td/table/tbody/tr[5]/td[4]//text()').extract_first()
        Facing = response.xpath('//*[@id="trandetail"]/table/tbody/tr/td/table/tbody/tr[7]/td[2]//text()').extract_first()
        Layout = response.xpath('//*[@id="trandetail"]/table/tbody/tr/td/table/tbody/tr[7]/td[4]//text()').extract_first()
        return OP_Date, Gross_Area, Exp_Year, Facing, Layout


    def parse_detail_page1(self, response):
        title = response.css('h1::text').extract()[0].strip()
        price = response.css('.pricelabel > strong::text').extract()[0]

        item = OlxItem()
        item['title'] = title
        item['price'] = price
        item['url'] = response.url
        yield item
