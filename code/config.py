"""Shared settings: paths, city names, case definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"

# Chinese city name (as it appears in the raw files) -> English name.
# Xiangyang was called Xiangfan until Dec 2010, so both names map to one city.
CITY_NAMES = {
    "北京": "Beijing", "上海": "Shanghai", "重庆": "Chongqing",
    "唐山": "Tangshan", "秦皇岛": "Qinhuangdao", "包头": "Baotou",
    "锦州": "Jinzhou", "吉林": "Jilin", "扬州": "Yangzhou", "蚌埠": "Bengbu",
    "安庆": "Anqing", "泉州": "Quanzhou", "九江": "Jiujiang", "赣州": "Ganzhou",
    "烟台": "Yantai", "济宁": "Jining", "洛阳": "Luoyang",
    "平顶山": "Pingdingshan", "宜昌": "Yichang", "襄阳": "Xiangyang",
    "襄樊": "Xiangyang", "岳阳": "Yueyang", "常德": "Changde", "桂林": "Guilin",
    "北海": "Beihai", "惠州": "Huizhou", "韶关": "Shaoguan", "南充": "Nanchong",
    "杭州": "Hangzhou", "合肥": "Hefei", "福州": "Fuzhou", "济南": "Jinan",
    "广州": "Guangzhou", "海口": "Haikou", "温州": "Wenzhou",
}

# Beijing candidate pool (paper Section 4.1, 21 cities without purchase restrictions).
BEIJING_POOL = [
    "Tangshan", "Qinhuangdao", "Baotou", "Jinzhou", "Jilin", "Yangzhou",
    "Bengbu", "Anqing", "Quanzhou", "Jiujiang", "Ganzhou", "Yantai", "Jining",
    "Luoyang", "Pingdingshan", "Yichang", "Xiangyang", "Yueyang", "Changde",
    "Guilin", "Beihai",
]

# One entry per policy case. Controls are the sets reported in the paper
# (Tables 2, 4, 6). Sample windows follow Sections 4.1-4.3.
CASES = {
    "beijing": dict(
        title="Beijing: home-purchase restriction", target="Beijing",
        controls=["Jilin", "Yangzhou", "Bengbu", "Anqing", "Pingdingshan",
                  "Xiangyang", "Guilin"],
        pre=("2008-01", "2010-04"), post=("2010-05", "2011-11")),
    "shanghai": dict(
        title="Shanghai: trial property tax", target="Shanghai",
        controls=["Hangzhou", "Hefei", "Fuzhou", "Jinan", "Guangzhou",
                  "Haikou", "Wenzhou"],
        pre=("2008-01", "2011-01"), post=("2011-02", "2012-02")),
    "chongqing": dict(
        title="Chongqing: trial property tax", target="Chongqing",
        controls=["Bengbu", "Anqing", "Quanzhou", "Ganzhou", "Pingdingshan",
                  "Yueyang", "Huizhou", "Shaoguan", "Guilin", "Nanchong"],
        pre=("2008-01", "2011-01"), post=("2011-02", "2012-11")),
}

# Plot colours (blue = actual, orange = counterfactual).
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a96"
