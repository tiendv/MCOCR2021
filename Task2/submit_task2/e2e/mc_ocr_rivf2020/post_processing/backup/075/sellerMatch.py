# !pip3 install python-levenshtein

from Levenshtein import *
import codecs
import unidecode
from .create_prices_proprocess_json import SELLER_PREPROCESS
from .create_prices_proprocess_json import PREFIX_PRIORITIZE


LIST_OUTPUT_PRICES = []
for key, value in PREFIX_PRIORITIZE.items():
	LIST_OUTPUT_PRICES.append(key)

with open("post_processing/field_dictionary/prices_prioritize.txt") as f:
    content = f.readlines()
LIST_PRICES_PRIORITIZE_DEF = [x.strip() for x in content] 

# print(LIST_OUTPUT_PRICES)

Sellers = [_.upper() for _ in SELLER_PREPROCESS]

def sellerMatch(raw_input):
	raw_input = raw_input.upper()
	index_min = max(range(len(Sellers)), \
		key=lambda x: ratio(raw_input, Sellers[x]))
	return SELLER_PREPROCESS[index_min]

def prefixMatch(raw_input):
	# raw_input = raw_input.upper()
	index_min = max(range(len(LIST_PRICES_PRIORITIZE_DEF)), \
		key=lambda x: ratio(raw_input, LIST_PRICES_PRIORITIZE_DEF[x]))
	return LIST_PRICES_PRIORITIZE_DEF[index_min] if distance(raw_input, LIST_PRICES_PRIORITIZE_DEF[index_min]) < 5 else None

def output_Prices_Match(raw_input):
	raw_input = raw_input.lower()
	index_min = max(range(len(LIST_OUTPUT_PRICES)), \
		key=lambda x: ratio(raw_input, LIST_OUTPUT_PRICES[x]))
	return PREFIX_PRIORITIZE[LIST_OUTPUT_PRICES[index_min]] if distance(raw_input, LIST_OUTPUT_PRICES[index_min]) < 5 else 999 # nếu sai nhiều bỏ cái if

# print(sellerMatch('Chợ sủi phú thị gia lâm 19 28    '))
# print(output_Prices_Match("Tổng tiền (VAT):"))