# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class DemoItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    Input_Date = scrapy.Field()
    Address = scrapy.Field()
    Floor = scrapy.Field()
    Unit = scrapy.Field()
    Saleable_Area = scrapy.Field()
    Price_M = scrapy.Field()
    Price_ft2 = scrapy.Field()
    OP_Date = scrapy.Field()
    Gross_Area = scrapy.Field()
    Exp_Year = scrapy.Field()
    Facing = scrapy.Field()
    Layout = scrapy.Field()
    pass
