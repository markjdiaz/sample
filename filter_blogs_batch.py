import sys, csv, os, glob


#This script runs through the .csv files of scraped blog content and finds posts that
#contain the specified keywords. The posts, along with their blog info are written to a
#new .csv file.

KEYWORDS = [' old ', 'young']
global KEYWORD_COUNTS
KEYWORD_COUNTS = []
global prev_date
prev_date = None
title_index = 1
date_index = 2
content_index = 3
blog_count = 0
total_post_count = 0
total_match_count = [0,0]

def keyword_in_post(post_content):
	for word in KEYWORDS:
		if word in post_content:
			return True
	return False

def find_matches(input_filename, output_writer):
	input_file = open(input_filename, 'rb')
	input_reader = csv.reader(input_file, dialect='excel')

	#summary_row = [input_filename]
	curr_post_date = None
	local_match_count = 0
	local_post_count = 0


	for i, input_row in enumerate(input_reader):
		if len(input_row) == 3:
			title_index = 0
			date_index = 1
			content_index = 2
		else:
			title_index = 1
			date_index = 2
			content_index = 3

		if input_row[date_index] != "":
			curr_post_date = input_row[date_index]
			prev_date = curr_post_date

		if keyword_in_post(input_row[content_index]):
			content = input_row[content_index]
			for i, keyword in enumerate(KEYWORDS):
				count = content.count(keyword)
				local_match_count = local_match_count + count

			input_row.insert(0, input_filename)
			output_writer.writerow(input_row)
			local_match_count += 1
		local_post_count += 1

	return (local_match_count, local_post_count)


output_filename = "death.csv"
output_file = open(output_filename, 'wb')
output_writer = csv.writer(output_file, dialect='excel')

for x in range(0,len(KEYWORDS)):
	KEYWORD_COUNTS.append(0)

blog_matches = 0
i = 0
for filename in glob.glob('*.csv'):
	if filename in IGNORE: continue
	result = find_matches(filename, output_writer)
	matches = result[0]
	if matches > 0: blog_matches += 1
	i += matches
print i