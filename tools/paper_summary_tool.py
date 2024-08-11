'''
'''

import os
import subprocess
from io import BytesIO
import shutil
import time
import json
from pathlib import Path

from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from prompt import summary_paper_prompt
from openai import OpenAI



ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def git_upload_file(commit_info):
    now_path = os.getcwd()
    directory_path = f'{ROOT_PATH}'
    # 改变当前工作目录
    os.chdir(directory_path)

    command = "git add ."  # 对于Unix/Linux
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    command = f'git commit -m "paper summary {commit_info}"'  # 对于Unix/Linux
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    command = f'git push'  # 对于Unix/Linux
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    os.chdir(now_path)

class SummaryPaper:
    def __init__(self, prompt) -> None:
        self.prompt = prompt
        self.model_name = "qwen-long"
        self.llm_client = OpenAI(
            api_key="sk-d71e4818164d4813b6fe6c9f70c2e745",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def __init_env(self, day):
        # data的格式为2024-08-01， 获取年和月份
        year, month, _ = day.split('-')
        # 以年和月组合创建文件夹
        folder_path = f'{ROOT_PATH}/{year}/{month}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(f'{folder_path}/paper_info'):
            os.makedirs(f'{folder_path}/paper_info')
        if not os.path.exists(f'{folder_path}/summary_info'):
            os.makedirs(f'{folder_path}/summary_info')
        return folder_path
    def set_day_info(self, day):
        self.hf_url = f'https://huggingface.co/papers?date={day}'
        root_path = self.__init_env(day)
        self.paper_info_file_name = f'{root_path}/paper_info/{day}.json'
        self.summary_info_file_name = f'{root_path}/summary_info/{day}.md' 


    def __get_hf_dailay_paper(self) -> str:
        response = requests.get(self.hf_url)
        return response.text
    
    def parse_hf_html(self):
        html_content = self.__get_hf_dailay_paper()
        soup = BeautifulSoup(html_content, "html.parser")
        a_tags = soup.find_all("a")
        paper_titles_and_links = []

        for tag in a_tags:
            title = tag.get_text(strip=True)
            link = tag.get("href")
            if title and link:
                paper_titles_and_links.append({"title": title, "link": link})
        # 过滤无效内容
        paper_item = []
        for item in paper_titles_and_links:
            if '/papers/2' in item['link'] and  len(item['title']) > 15:
                paper_item.append(item)
        for item in paper_item:
            link = item['link']
            arxiv_id = link.split('/')[-1]
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
            arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            item['arxiv_pdf_url'] = arxiv_pdf_url
            item['arxiv_url'] = arxiv_url
            item['arxiv_id'] = arxiv_id
        return paper_item 
    def __save_to_json(self, data, file_path):
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    
    def download_pdf(self, url, path, local_filename=None):
        """
        下载指定URL的PDF文件并保存到本地。

        :param url: PDF文件的网址
        :param local_filename: 本地保存的文件名，默认为None则使用URL的最后一部分作为文件名
        """
        # 发送HTTP GET请求获取PDF文件内容
        response = requests.get(url, stream=True)

        # 检查请求是否成功
        if response.status_code == 200:
            # 如果未指定本地文件名，则从URL中提取
            if not local_filename:
                content_disposition = response.headers.get('content-disposition')
                if content_disposition:
                    filename = content_disposition.split("filename=")[-1]
                    local_filename = filename.strip("\"'")
                else:
                    local_filename = url.split("/")[-1]

            # 使用BytesIO处理二进制数据，避免大文件内存问题
            with BytesIO(response.content) as pdf_buffer:
                # 使用shutil将BytesIO对象的内容写入本地文件
                with open(os.path.join(path, local_filename), 'wb') as f:
                    shutil.copyfileobj(pdf_buffer, f)

            print(f"PDF文件已成功保存为: {local_filename}")
        else:
            print(f"请求失败，状态码：{response.status_code}")
        return os.path.join(path, local_filename)

    def __summary_paper(self, pdf_file_path):
        file_object = self.llm_client.files.create(file=Path(pdf_file_path), purpose="file-extract")
        print(f'{pdf_file_path} create finish: {file_object}')
        return self.__qwen_llm_handle(file_object)
    
    def __qwen_llm_handle(self, file_object):
        completion = self.llm_client.chat.completions.create(
        model = self.model_name,
        messages = [
            {
                'role': 'system',
                'content': self.prompt,
            },
            {
                'role': 'system',
                'content': f'fileid://{file_object.id}'
            },
            {
                'role': 'user',
                'content': '请总结：'
            }
        ],
            stream=True
        )
        content = []
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content.append(chunk.choices[0].delta.content)
                # print(chunk.choices[0].dict())
        return ''.join(content)

    def summary_paper(self, pdf_file_path, retry = 5) -> str:
        for i in range(retry):
            try:
                summary_info = self.__summary_paper(pdf_file_path)
                return summary_info
            except Exception as e:
                print(f"summary_paper_retry error: {e}")
                time.sleep(10)

        return '# faild to  read !!!!'


    def summary_papers(self, day) -> str:
        self.set_day_info(day)
        paper_info = self.parse_hf_html()
        self.__save_to_json(paper_info, self.paper_info_file_name)
        for paper in tqdm(paper_info):
            file_path = self.download_pdf(paper['arxiv_pdf_url'], '/tmp')
            summary_info = self.summary_paper(file_path)
            os.remove(file_path)
            with open(self.summary_info_file_name, 'a', encoding='utf-8') as file:
                file.write("\n")
                file.write(f'# {paper["title"]}\n')
                file.write(f'[arxiv_pdf_url]({paper["arxiv_pdf_url"]})\n')
                file.write(summary_info)

def test(day):
    summary = SummaryPaper(summary_paper_prompt)
    summary.set_day_info(day)
    # summary.download_pdf('https://arxiv.org/pdf/2407.21783', '/tmp')
    pdf_path = '/tmp/2407.21783v1.pdf'
    summary.summary_paper(pdf_path)

if __name__ == '__main__':
    days = ['2024-08-05', '2024-08-06', '2024-08-07', '2024-08-08', '2024-08-09']
    # summary = SummaryPaper(summary_paper_prompt)
    # for day in days:
    #     summary.summary_papers(day)
    git_upload_file(','.join(days))