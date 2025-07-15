
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_helper import llm


def preprocess(raw_file_path, processed_file_path= "data/preprocessed.json"):
    enriched_posts =[]
    with open(raw_file_path, encoding ='utf-8') as json_file:
        data = json.load(json_file)
        for post in data:
            metadata = extract_metadata(post)
            mid_post = post | metadata
            enriched_posts.append(mid_post)
        for epost in enriched_posts:
            print (epost)
    unified_tags = get_unified_tag(enriched_posts)
    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags[tag] for tag in current_tags}
        post['tags'] = list(new_tags)

    with open(processed_file_path, encoding='utf-8', mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)

def get_unified_tag(enriched_posts):
    unique_tag = set()
    for post in enriched_posts:
        unique_tag.update(post['tags'])
    unique_tags_list = ', '.join(unique_tag)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
        1. Tags are unified and merged to create a shorter list. 
           Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
           Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
           Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
           Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
        2. Each tag should be follow title case convention. example: "Motivation", "Job Search"
        3. Output should be a JSON object, No preamble
        3. Output should have mapping of original tag and the unified tag. 
           For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation}}

        Here is the list of tags: 
        {tags}
        '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": str(unique_tags_list)})
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res

def extract_metadata(post):
    template = '''
        You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
        1. Return a valid JSON. No preamble. 
        2. JSON object should have exactly three keys: line_count, language and tags. 
        3. tags is an array of text tags. Extract maximum two tags.
        4. Language should be English or Hinglish (Hinglish means hindi + english)

        Here is the actual post on which you need to perform this task:  
        {post}
        '''
    pt = PromptTemplate.from_template(template)  #just read the template and get to know that there is some place holder called post
    chain = pt | llm  #create chain which will call llm
    response = chain.invoke(input={"post": post}) #invoking the chain where we are inserting post in place of post

    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)  #make the llm output in dictoinary
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res


if __name__ == "__main__":
    preprocess("data/raw_posts.json","data/preprocessed.json")