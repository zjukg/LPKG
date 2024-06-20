import pandas as pd
import re
import argparse
from retrieval import Retriever
from gpt.call_gpt import request_with_try
from tqdm import tqdm
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    # required=True,
    type=str,
    default=None,
    help=".json file containing question and answers, similar format to reader data",
)
parser.add_argument("--passages", type=str, default='./wiki_dump/psgs_w100.tsv', help="Path to passages (.tsv file)")
parser.add_argument("--passages_embeddings", type=str, default='./wiki_dump/wikipedia_embeddings/*', help="Glob path to encoded passages")
parser.add_argument(
    "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
)
parser.add_argument("--n_docs", type=int, default=5, help="Number of documents to retrieve per questions")
parser.add_argument(
    "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
)
parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
parser.add_argument(
    "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
)
parser.add_argument(
    "--model_name_or_path", type=str, default='./contriever_ms', help="path to directory containing model weights and config file"
)
parser.add_argument("--no_fp16", default=False, help="inference in fp32")
parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
parser.add_argument(
    "--indexing_batch_size", type=int, default=100000000, help="Batch size of the number of passages indexed"
)
parser.add_argument("--projection_size", type=int, default=768)
parser.add_argument(
    "--n_subquantizers",
    type=int,
    default=0,
    help="Number of subquantizer used for vector quantization, if 0 flat index is used",
)
parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
parser.add_argument("--lang", nargs="+")
parser.add_argument("--dataset", type=str, default="none")
parser.add_argument("--lowercase", default=True, help="lowercase text before encoding")
parser.add_argument("--normalize_text", default=True, help="normalize text")

args = parser.parse_args()

retriever = Retriever(args=args)
retriever.setup_retriever()

# query = 'What is the height of the player who won the 2015 AFL Rising Star award?'
# print(retriever.search_document(query=query,top_n=5))


def llm(prompt:str):
    return request_with_try(prompt)
def retrieve(question:str):
    passages = retriever.search_document(query=question,top_n=5)
    info = str()
    for idx,p in enumerate(passages):
        text = f'# Passage {idx+1}\n'
        text += 'Title:' + p['title'] +'\nText:'+p['text'] + '\n'
        info += text
    return info


class QA_Pipeline:
    def __init__(self,original_q) -> None:
        self.original_q = original_q
        self.q1 = None
        self.info1 = None
        self.ans1 = None
        self.q2 = None
        self.info2 = None
        self.ans2 = None
        self.q3 = None
        self.info3 = None
        self.ans3 = None
        self.inter1_result = None
        self.inter2_result = None
        self.union1_result = None
        self.error_flag = False
        self.final_ans = None
        self.strategy = False
    def sub_QA_prompt(self,question:str,info:str):
        prompt = f"""Given a question and some information that may help you answer the question. Please answer the question based on your own knowledge and the information provided. 
### Information:
{info}
### Question:
{question}
### Your Answer: (You only need to provide the final answer of the question. Intermediate answers are not needed. Please return your answer in the form of a list, where each element in the list is a short entity answer, such as [Apple]. When you think there are multiple answers, please divide them with a '#' symbol, such as [Apple#Banana#Origin]. If the answer is not included in the information provided, please answer based on your own knowledge. If you don't know either, please return [None].)
"""
        return prompt
    def sub_QA_prompt_v2(self,question:str,info:str):
        prompt = f"""Given a question and some information that may help you answer the question. Please answer the question based on your own knowledge or the information provided. 
### Information:
{info}
### Question:
{question}
### Your Answer: (Important!! You should only return [Yes] or [No], other content are not required)
"""
        return prompt
    def direct_prompt(self,question:str):
        prompt = f"{question} Your answer should only be one word or phrase. If you think there are multiple answers, return a list like [Answer1#Answer2#..] which you should use the symbol '#' to separate the different answers."
        return prompt
    def direct_answer(self,question:str):
        prompt = self.direct_prompt(question)
        sc_map = {}
        for i in range(5):
            ans = llm(prompt)
            if(ans not in sc_map.keys()):
                sc_map[ans] = 1
            else:
                sc_map[ans] += 1
        max_value = max(sc_map.values())
        max_keys = [key for key, value in sc_map.items() if value == max_value]
        sc_ans = random.choice(max_keys)
        
        return sc_ans.strip()

    def subquestion_answer(self,question:str):
        infomation = retrieve(question)
        if(not self.strategy):
            prompt = self.sub_QA_prompt(question,infomation)
        else:
            prompt = self.sub_QA_prompt_v2(question,infomation)
        # import pdb;pdb.set_trace()
        sc_map = {}
        for i in range(5):
            ans = llm(prompt)
            if(ans not in sc_map.keys()):
                sc_map[ans] = 1
            else:
                sc_map[ans] += 1
        max_value = max(sc_map.values())
        max_keys = [key for key, value in sc_map.items() if value == max_value]
        sc_ans = random.choice(max_keys)
                
        return sc_ans.strip(),infomation

    def compare_QA_prompt(self,sub_q_list:list,sub_ans_list:list):
        instruction = "Your task is to answer a comparison question. I will first tell you the original question, and tell you some sub-questions and their corresponding answers to the sub-questions. Please answer the original comparison question based on the sub-questions and the answers to the sub-questions."
        prompt = f"{instruction}\n### Original Question:\n{self.original_q}"
        for i in range(len(sub_q_list)):
            prompt += f"### Sub-Question {i+1}:\n{sub_q_list[i]}\n### Answer to Sub-Question {i+1}:\n{sub_ans_list[i]}"
        prompt += f"### Please Answer the Original Question: {self.original_q} (You only need to provide the final answer of the question. Intermediate answers are not needed. Please return your answer in the form of a list, where each element in the list is a short entity answer, such as [Yes] or [No] or [Entity], etc. If the answer is not included in the information provided, please answer based on your own knowledge. If you don't know either, please return [None].)"
        return prompt
    
    def compare_QA_prompt_v2(self,sub_q_list:list,sub_ans_list:list):
        instruction = "Your task is to answer a comparison question. I will first tell you the original question, and tell you some sub-questions and their corresponding answers to the sub-questions. Please answer the original comparison question based on the sub-questions and the answers to the sub-questions."
        prompt = f"{instruction}\n### Original Question:\n{self.original_q}\n"
        for i in range(len(sub_q_list)):
            prompt += f"### Sub-Question {i+1}:\n{sub_q_list[i]}\n### Answer to Sub-Question {i+1}:\n{sub_ans_list[i]}"
        prompt += f"### Please Answer the Original Question: {self.original_q} (Important!! You should only return [Yes] or [No], other content are not required)"
        return prompt

    def compare_QA(self,sub_q_list:list,sub_ans_list:list):
        if(not self.strategy):
            prompt = self.compare_QA_prompt(sub_q_list,sub_ans_list)
        else:
            prompt = self.compare_QA_prompt_v2(sub_q_list,sub_ans_list)
        # import pdb;pdb.set_trace()
        ans = llm(prompt)
        return ans

    def str2set(self,s:str):
        s = s.strip()
        ans_list = s.lstrip('[').rstrip(']').split('#')
        if(isinstance(ans_list, str)):
            ans_list = [ans_list]
        for i in range(len(ans_list)):
            a = ans_list[i].strip()
            a = a.lstrip('[').rstrip(']')
            ans_list[i] = a
        # if(len(ans_list)>5):
        #     ans_list = ans_list[0:5]
        ans_set = set(ans_list)
        ans_set.discard('None')
        return ans_set


    def Intersection(self,inter_call:str):
        answer_set = set()
        if('Ans_1' in inter_call):
            ans_1 = self.ans1
            ans_1 = ans_1[0]
            if(ans_1):
                answer_set = ans_1
        if('Ans_2' in inter_call):
            ans_2 = set()
            if (self.ans2):
                for s in self.ans2:
                    if(s):
                        ans_2 = ans_2.union(s)
            if(len(answer_set)==0):
                answer_set = ans_2
            else:
                answer_set = answer_set.intersection(ans_2)
        if('Inter_Results1' in inter_call):
            if(len(answer_set)==0):
                answer_set = self.inter1_result
            else:
                answer_set = answer_set.intersection(self.inter1_result)
        # import pdb;pdb.set_trace()
        if('Ans_3' in inter_call):
            ans_3 = set()
            if(self.ans3):
                for s in self.ans3:
                    if(s):
                        ans_3 = ans_3.union(s)
            if(len(answer_set)==0):
                answer_set = ans_3
            else:
                answer_set = answer_set.intersection(ans_3)
        # import pdb;pdb.set_trace()
        if('Inter_Results1' not in inter_call):
            self.inter1_result = answer_set
        else:
            self.inter2_result = answer_set
        
    def Union(self,union_call:str):
        answer_set = set()
        if('Ans_1' in union_call):
            ans_1 = self.ans1
            ans_1 = ans_1[0]
            if(ans_1):
                answer_set = ans_1
        if('Ans_2' in union_call):
            ans_2 = set()
            if (self.ans2):
                for s in self.ans2:
                    if(s):
                        ans_2 = ans_2.union(s)
            if(len(answer_set)==0):
                answer_set = ans_2
            else:
                answer_set = answer_set.union(ans_2)
        if('Ans_3' in union_call):
            ans_3 = set()
            if (self.ans3):
                for s in self.ans3:
                    if(s):
                        ans_3 = ans_3.union(s)
            if(len(answer_set)==0):
                answer_set = ans_3
            else:
                answer_set = answer_set.union(ans_3)
        self.union1_result = answer_set
    
    def Compare(self,compare_call:str,compare_store:str):
        compare_sub_q = []
        compare_sub_ans = []
        if('Ans_1' in compare_call and self.q1 and self.ans1):
            compare_sub_q.extend(self.q1)
            compare_sub_ans.extend(self.ans1)
        if('Ans_2' in compare_call and self.q2 and self.ans2):
            compare_sub_q.extend(self.q2)
            compare_sub_ans.extend(self.ans2)
        if('Ans_3' in compare_call and self.q3 and self.ans3):
            compare_sub_q.extend(self.q3)
            compare_sub_ans.extend(self.ans3)
        compare_results = self.compare_QA(compare_sub_q,compare_sub_ans)
        print(compare_results)
        compare_results = [self.str2set(compare_results)]
        if('Ans_1' in compare_store):
            self.ans1 = compare_results
            print('Save answer to Ans_1')
        if('Ans_2' in compare_store):
            self.ans2 = compare_results
            print('Save answer to Ans_2')
        if('Ans_3' in compare_store):
            self.ans3 = compare_results
            print('Save answer to Ans_3')

            
    def Final_QA(self,final_call:str):
        if('Ans_1' in final_call):
            answer = self.ans1
            self.final_ans = answer[0]
        elif('Ans_2' in final_call):
            answer = set()
            if(self.ans2):
                for s in self.ans2:
                    if(s):
                        answer = answer.union(s)
            self.final_ans = answer
        elif('Ans_3' in final_call):
            answer = set()
            if(self.ans3):
                for s in self.ans3:
                    if(s):
                        answer = answer.union(s)
            self.final_ans = answer
        elif('Inter_Results1' in final_call):
            self.final_ans = self.inter1_result
        elif('Inter_Results2' in final_call):
            self.final_ans = self.inter2_result
        elif('Union_Results1' in final_call):
            self.final_ans = self.union1_result
        
    
    def output_parse(self,llm_predict:str):
        predict_lines = llm_predict.split('\n')

        # 当中间某个子问题，LLM无法回答时，记为True，调用保底
        error_flag = False

        for line in predict_lines:
            # import pdb;pdb.set_trace()
            if('Sub_Question_1: str = ' in line):
                q1 = line.split('Sub_Question_1: str = ')[1].lstrip('f').strip("\"")
                print('Sub_Question_1:'+q1)
                self.q1 = [q1]
                ans1,info = self.subquestion_answer(q1)
                self.ans1 = [self.str2set(ans1)]
                # self.ans1.discard('None')
                self.info1 = [info]
                print('Ans_1:'+str(self.ans1))
                if(len(self.ans1)==0):
                    error_flag = True
                    break
            if('Sub_Question_2: str = ' in line):
                q2 = line.split('Sub_Question_2: str = ')[1].lstrip('f').strip("\"")
                print('Sub_Question_2:'+q2)
                # answer2 = set()
                question2 = []
                info2 = []
                answer2 = []
                if("{Ans_1}" in q2):
                    ans_1 = set()
                    for s in self.ans1:
                        if(s):
                            ans_1 = ans_1.union(s)
                    for idx,a in enumerate(ans_1):
                        q2_1 = q2.replace("{Ans_1}",a)
                        print('Sub_Question_2_' + str(idx) + ':' +q2_1)
                        ans2,info = self.subquestion_answer(q2_1)
                        # answer2 = answer2.union(self.str2set(ans2))
                        answer2.append(self.str2set(ans2))
                        print('Ans_2_'+ str(idx) + ':' + str(self.str2set(ans2)))
                        question2.append(q2_1)
                        info2.append(info)
                else:
                    ans2,info = self.subquestion_answer(q2)
                    # answer2 = answer2.union(self.str2set(ans2))
                    answer2.append(self.str2set(ans2))
                    print('Ans_2:' + str(self.str2set(ans2)))
                    question2.append(q2)
                    info2.append(info)
                self.q2 = question2
                # answer2.discard('None')
                self.ans2 = answer2
                self.info2 = info2
                print('Sub_Question2:' + q2)
                print('All Ans_2:' + str(self.ans2))
                if(len(self.ans2)==0):
                    error_flag = True
                    break

            if('Sub_Question_3: str = ' in line):
                q3 = line.split('Sub_Question_3: str = ')[1].lstrip('f').strip("\"")
                print('Sub_Question_3:'+q3)
                # answer3 = set()
                question3 = []
                info3 = []
                answer3 = []
                ans_1 = set()
                for s in self.ans1:
                    if(s):
                        ans_1 = ans_1.union(s)
                ans_2 = set()
                for s in self.ans2:
                    if(s):
                        ans_2 = ans_2.union(s)
                if("{Ans_1}" in q3):
                    for idx1,a in enumerate(ans_1):
                        q3_1 = q3.replace("{Ans_1}",a)
                        if("{Ans_2}" in q3_1):
                            for idx2,b in enumerate(ans_2):
                                q3_2 = q3_1.replace("{Ans_2}",b)
                                print('Sub_Question_3_' + str(idx1) + '_' + str(idx2) + ':' +q3_2)
                                ans3,info = self.subquestion_answer(q3_2)
                                # answer3 = answer3.union(self.str2set(ans3))
                                answer3.append(self.str2set(ans3))
                                print('Ans_3_'+ str(idx1) + '_' + str(idx2) + ':' + str(self.str2set(ans3)))
                                question3.append(q3_2)
                                info3.append(info)
                        else:
                            print('Sub_Question_3_' + str(idx1) + ':' +q3_1)
                            ans3,info = self.subquestion_answer(q3_1)
                            # answer3 = answer3.union(self.str2set(ans3))
                            answer3.append(self.str2set(ans3))
                            print('Ans_3_'+ str(idx1) + ':' + str(self.str2set(ans3)))
                            question3.append(q3)
                            info3.append(info)
                elif("{Ans_2}" in q3):
                    for idx,b in enumerate(ans_2):
                        q3_1 = q3.replace("{Ans_2}",b)
                        print('Sub_Question_3_' + str(idx) + ':' +q3_1)
                        ans3,info = self.subquestion_answer(q3_1)
                        info3.append(info)
                        question3.append(q3_1)
                        # answer3 = answer3.union(self.str2set(ans3))
                        answer3.append(self.str2set(ans3))
                        print('Ans_3_'+ str(idx) + ':' + str(self.str2set(ans3)))
                else:
                    # print('Sub_Question_3:' +q3)
                    ans3,info = self.subquestion_answer(q3)
                    # answer3 = answer3.union(self.str2set(ans3))
                    answer3.append(self.str2set(ans3))
                    print('Ans_3:' + str(self.str2set(ans3)))
                    question3.append(q3)
                    info3.append(info)
                self.q3 = question3
                self.info3 = info3
                # answer3.discard('None')

                self.ans3 = answer3
                print('Sub_Question3:' + q3)
                print('All Ans_3:' + str(self.ans3))
                if(len(self.ans3)==0):
                    error_flag = True
                    break

            if('Inter_Results1: str = Intersection' in line):
                inter_call = line.split('Inter_Results1: str = Intersection')[1]
                print(line)
                self.Intersection(inter_call)
                print('Results:' + str(self.inter1_result))
                if(len(self.inter1_result)==0):
                    error_flag = True
                    break
            if('Inter_Results2: str = Intersection' in line):
                inter_call = line.split('Inter_Results2: str = Intersection')[1]
                print(line)
                self.Intersection(inter_call)
                print('Results:' + str(self.inter2_result))
                if(self.inter2_result==None or len(self.inter2_result)==0):
                    error_flag = True
                    break
            if('Union_Results1: str = Union' in line):
                union_call = line.split('Union_Results1: str = Union')[1]
                print(line)
                self.Union(union_call)
                print(self.union1_result)
                if(len(self.union1_result)==0):
                    error_flag = True
                    break
            if('str = Compare' in line):
                compare_call = line.split('Compare')[1].lstrip("(").rstrip(")")
                print(line)
                compare_store = line.split('str = Compare')[0]
                self.Compare(compare_call,compare_store)
                

            if('Final_Answer: str = ' in line):
                if('Answer = ' in line):
                    final_call = line.split('Answer = ')[1].strip(")")
                    self.Final_QA(final_call)
                else:
                    error_flag = True
                    break


        # import pdb;pdb.set_trace()
        self.error_flag = error_flag
        if(self.final_ans is None or len(self.final_ans) == 0 or error_flag==True):
            print('Error. Answer Question directly.')
            final_ans,info = self.subquestion_answer(self.original_q)
            # final_ans = self.direct_answer(self.original_q)
            self.final_ans = self.str2set(final_ans)
            print('Final Answer:'+str(self.final_ans))
            return self.final_ans
        else:
            print('Final Answer:'+str(self.final_ans))
            return self.final_ans
        
# predict = """
# ### Question Type: Compare
# ### Decompose the original question into sub-questions.

# Thought1: str = "If I want to know which magazine was started first, I need to first know when Arthur's Magazine was started."
# Sub_Question_1: str = "When was Arthur's Magazine started?"
# Info_1: str = Search(query = Sub_Question_1, thought = Thought1)
# Ans_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)

# Thought2: str = "At the same time, I need to know when First for Women was started."
# Sub_Question_2: str = "When was First for Women started?"
# Info_2: str = Search(query = Sub_Question_2, thought = Thought2)
# Ans_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)


# Thought3: str = "After knowing when Arthur's Magazine was started (i.e., Ans_1) and when First for Women was started, I need to compare the two dates."
# Ans_3: str = Compare(Original_Query = Original_Question, Subquestions = [Sub_Question_1,Sub_Question_2], Answers = [Ans_1,Ans_2])

# Final_Answer: str = Finish_The_Plan(Answer = Ans_3)
# """

# pipeline = QA_Pipeline("Which magazine was started first Arthur's Magazine or First for Women?")
# result = pipeline.output_parse(predict)
# print(result)
import csv
import os

def get_row_count(filename):
    try:
        with open(filename, "r", newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            return sum(1 for row in reader)
    except FileNotFoundError:
        # 如果文件不存在，返回0
        return 1

def write_to_csv_with_headers(filename, data, headers, mode='w'):
    with open(filename, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(headers)
        writer.writerows(data)

plan_result  = '' # Your planning result (a jsonl file, each line represents a planning result)
plan = []
question = []
labels = []
with open(plan_result,'r',encoding='utf-8') as f:
    for line in f:
        d = json.loads(line.strip())
        plan.append(d['predict'])
        question.append(d['question'])
        labels.append(d['label'])
# import pdb;pdb.set_trace()
l = len(question)
results = []

result_file = '' # path of final results (csv file)
headers = ['Plan','Original Q','Q1','Info1','Ans1','Q2','Info2','Ans2','Q3','Info3','Ans3','Inter1_result','Inter2_result','Union1_result','Error_flag','Final Answer','Label']
start = get_row_count(result_file)
cur = start-1


def parse_plan(i):
    p,q = plan[i],question[i]
    pipeline = QA_Pipeline(q)
    if('strategy' in result_file):
        pipeline.strategy = True
    answer = pipeline.output_parse(p)
    label = labels[i]
    return [p,
            pipeline.original_q,
            pipeline.q1,
            pipeline.info1,
            pipeline.ans1,
            pipeline.q2,
            pipeline.info2,
            pipeline.ans2,
            pipeline.q3,
            pipeline.info3,
            pipeline.ans3,
            pipeline.inter1_result,
            pipeline.inter2_result,
            pipeline.union1_result,
            pipeline.error_flag,
            pipeline.final_ans,
            label]

with ThreadPoolExecutor(max_workers=10) as executor:
    # 将 future 与对应索引 i 关联起来
    for j in range(start-1,l,10):
        future_to_index = {executor.submit(parse_plan, i): i for i in range(j,min(j+10,l))}
        results = [None] * min(10,l-j)  # 创建与 original_q 相同长度的列表占位
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (i, exc))
            else:
                results[i-j] = data  # 根据索引 i 将数据填充到正确的位置
        mode = 'a' if os.path.exists(result_file) else 'w'
        write_to_csv_with_headers(result_file, results, headers, mode=mode)

# for p,q in tqdm(zip(plan[start-1:l],question[start-1:l]),total=l-cur):
#     if(cur%10==0):
#         mode = 'a' if os.path.exists(result_file) else 'w'
#         write_to_csv_with_headers(result_file, results, headers, mode=mode)
#         results.clear() # 清空列表用于收集下一批元素
#     pipeline = QA_Pipeline(q)
#     if('strategy' in result_file):
#         pipeline.strategy = True
#     answer = pipeline.output_parse(p)
#     label = labels[cur]
#     results.append([p,
#                     pipeline.original_q,
#                     pipeline.q1,
#                     pipeline.info1,
#                     pipeline.ans1,
#                     pipeline.q2,
#                     pipeline.info2,
#                     pipeline.ans2,
#                     pipeline.q3,
#                     pipeline.info3,
#                     pipeline.ans3,
#                     pipeline.inter1_result,
#                     pipeline.inter2_result,
#                     pipeline.union1_result,
#                     pipeline.error_flag,
#                     pipeline.final_ans,
#                     label])
#     cur += 1

# if(len(results)>0):
#     mode = 'a' if os.path.exists(result_file) else 'w'
#     write_to_csv_with_headers(result_file, results, headers, mode=mode)
#     results.clear()

    
# results_df = pd.DataFrame(results,columns=['Plan','Original Q','Q1','Info1','Ans1','Q2','Info2','Ans2','Q3','Info3','Ans3','Inter1_result','Inter2_result','Union1_result','Error_flag','Final Answer'])
# results_df.to_csv('test_results/hotpotqa_100.csv')