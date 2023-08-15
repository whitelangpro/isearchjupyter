import re
from typing import List

from langchain.text_splitter import CharacterTextSplitter


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        pre_txt = ""
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "。")
        sent_sep_pattern = re.compile(
            '([■●•﹒﹔﹖﹗．。！？]|\d{1,3}-\d{1,3}\.|$)') 
        pre_sep_pattern = re.compile('([■●•]|\d{1,3}-\d{1,3}\.)') 
        stop_pattern = re.compile('.*→P\..*')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if stop_pattern.match(ele) :
                pre_txt = ""
            elif pre_sep_pattern.match(ele) :
                pre_txt = ele
            elif sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(re.sub("[■●•]", "", pre_txt+ele))
                pre_txt = ""
        return sent_list
