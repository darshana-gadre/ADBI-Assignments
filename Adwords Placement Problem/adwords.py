import sys
import pandas as pd
import math
import random
import copy

random.seed(0)

# check_budget: returns 0 if budget exceeded, 1 otherwise

def check_budget(bids, dict_budget):
	advertisers = bids.keys()
	for advertiser in advertisers:
		if dict_budget[advertiser] >= bids[advertiser]:
			return 1
	return 0
	

# greedy algorithm implementation

def greedy(dict_budget, dict_bids, queries):
	
	revenue = 0.0
	
	for query in queries:
		
		advertisers = list(dict_bids[query].keys())
		
		max_bidder = advertisers[0]
		max_bid = -1
		
		temp = check_budget(dict_bids[query], dict_budget)
		
		if temp == 0:
			max_bidder = -1;
		
		if max_bidder != -1:
			
			for advertiser in advertisers:
				if dict_budget[advertiser] >= dict_bids[query][advertiser]:
					if max_bid < dict_bids[query][advertiser]:
						max_bidder = advertiser
						max_bid = dict_bids[query][advertiser]
					elif max_bid == dict_bids[query][advertiser]:
						if max_bidder > advertiser:
							max_bidder = advertiser
							max_bid = dict_bids[query][advertiser]
		
		
			revenue += dict_bids[query][max_bidder]
			dict_budget[max_bidder] -= dict_bids[query][max_bidder]
	
	return revenue
	

# balance algorithm implementation
	
def balance(dict_budget, dict_bids, queries):
	
	revenue = 0.0
	
	for query in queries:
		
		advertisers = list(dict_bids[query].keys())
		
		max_bidder = advertisers[0]
		
		temp = check_budget(dict_bids[query], dict_budget)
		
		if temp == 0:
			max_bidder = -1;
			
		if max_bidder != -1:
			
			for advertiser in advertisers:
				if dict_budget[advertiser] >= dict_bids[query][advertiser]:
					if dict_budget[max_bidder] < dict_budget[advertiser]:
						max_bidder = advertiser
					elif dict_budget[max_bidder] == dict_budget[advertiser]:
						if max_bidder > advertiser:
							max_bidder = advertiser
		
		
			revenue += dict_bids[query][max_bidder]
			dict_budget[max_bidder] -= dict_bids[query][max_bidder]
	
	return revenue
	

# calculating scaled bid for msvv algorithm

def scaled_bid(bid, budget, remaining_budget):
	xu = (budget - remaining_budget)/budget
	
	psi_xu = 1 - math.exp(xu-1)
	
	sb = bid*psi_xu
	
	return sb


# msvv algorithm implementation

def msvv(dict_budget, remaining_budgets, dict_bids, queries):
	
	for query in queries:
		
		advertisers = list(dict_bids[query].keys())
		
		max_scaled_bid=0.00
		max_bid=0.00
		max_bidder = -1
		
		for advertiser in advertisers:
			
			budget=dict_budget[advertiser]
			left=remaining_budgets[advertiser]
			
			if left < dict_bids[query][advertiser]:
				continue
				
			sb_current = scaled_bid(dict_bids[query][advertiser], dict_budget[advertiser], remaining_budgets[advertiser])
			
			if sb_current > max_scaled_bid:
				max_scaled_bid = sb_current
				max_bidder = advertiser
				max_bid = dict_bids[query][advertiser]
		
		if max_bidder != -1:
			remaining_budgets[max_bidder] -= max_bid
	
	revenue = 0.0
	
	for bidder in remaining_budgets:
		revenue += (dict_budget[bidder] - remaining_budgets[bidder])
	
	return revenue


# main

if sys.argv[1] == 'greedy':
	algo = 'greedy'
elif sys.argv[1] == 'balance':
	algo = 'balance'
elif sys.argv[1] == 'msvv':
	algo = 'msvv'
else:
	print('Invalid Input!')
	

# Processing Input	

df = pd.read_csv('bidder_dataset.csv')

dict_budget = {}
dict_bids = {}

for i in range(0, len(df)):
	advertiser = df.iloc[i]['Advertiser']
	keyword = df.iloc[i]['Keyword']
	bid_value = df.iloc[i]['Bid Value']
	budget = df.iloc[i]['Budget']
	
	if advertiser not in dict_budget:
		dict_budget[advertiser] = budget
	
	if keyword not in dict_bids:
		dict_bids[keyword] = {}
	
	if advertiser not in dict_bids[keyword]:
		dict_bids[keyword][advertiser] = bid_value
		

with open('queries.txt') as f:
	queries = f.readlines()
	
queries = [x.strip() for x in queries]


# calculating revenue

if algo == 'greedy':
   	ans = greedy(copy.deepcopy(dict_budget), copy.deepcopy(dict_bids), queries)
elif algo == 'balance':
   	ans = balance(copy.deepcopy(dict_budget), copy.deepcopy(dict_bids), queries)
elif algo == 'msvv':
   	ans = msvv(copy.deepcopy(dict_budget), copy.deepcopy(dict_budget), copy.deepcopy(dict_bids), queries)


# calculating competitive ratio

optimal = sum(dict_budget.values())
revenues = []

for i in range(100):
    random.shuffle(queries)
    
    if algo == 'greedy':
    	revenue = greedy(copy.deepcopy(dict_budget), copy.deepcopy(dict_bids), queries)
    elif algo == 'balance':
    	revenue = balance(copy.deepcopy(dict_budget), copy.deepcopy(dict_bids), queries)
    elif algo == 'msvv':
    	revenue = msvv(copy.deepcopy(dict_budget), copy.deepcopy(dict_budget), copy.deepcopy(dict_bids), queries)
    
    revenues.append(revenue)


avg_revenue = sum(revenues) / len(revenues)
competitive_ratio = avg_revenue / optimal

ans = round(ans, 2)
competitive_ratio = round(competitive_ratio, 2)

print('Revenue : ', format(ans,'.2f'))
print('Competitive Ratio : ', format(competitive_ratio,'.2f'))
