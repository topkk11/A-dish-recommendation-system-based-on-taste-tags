# 基于口味标签的菜品推荐系统

从用户评论中提取口味关键词，构建菜品画像和用户偏好画像，通过 TF-IDF 向量化和余弦相似度实现个性化菜品推荐。

## 功能

- **关键词提取**：使用 jieba 分词从评论中提取描述性口味关键词（TF-IDF / TextRank）
- **情感分析**：使用 SnowNLP 对关键词所在句子进行情感打分（0~1）
- **菜品画像**：按菜品聚合关键词频率与情感，构建口味标签云
- **用户画像**：按用户聚合历史评论关键词，构建偏好标签云
- **相似度推荐**：TF-IDF 向量化后计算余弦相似度，返回 Top-N 推荐
- **CSV 导入**：通过 Web 页面上传评论 CSV，一键触发全流程处理

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python 3 + Flask 3.x |
| 前端 | Bootstrap 5 CDN + Jinja2 服务端渲染 |
| 中文分词 | jieba |
| 情感分析 | SnowNLP |
| 向量化 | scikit-learn TfidfVectorizer |
| 数据处理 | pandas |
| 数据存储 | Pandas DataFrame → Pickle 文件（原型阶段） |

## 项目结构

```
├── run.py                    # 入口：python run.py
├── requirements.txt          # Python 依赖
├── config.py                 # NLP 参数配置
├── app/
│   ├── __init__.py           # Flask app 工厂
│   ├── views.py              # 路由处理
│   ├── nlp/
│   │   ├── keyword.py        # jieba 分词 + 关键词提取
│   │   ├── sentiment.py      # SnowNLP 情感分析
│   │   ├── matcher.py        # TF-IDF 向量化 + 余弦相似度
│   │   └── pipeline.py       # 全流程编排
│   ├── services/
│   │   ├── dish_service.py   # 菜品画像构建
│   │   ├── user_service.py   # 用户偏好画像构建
│   │   └── recommend_service.py  # 推荐计算
│   ├── data/
│   │   └── store.py          # Pickle I/O
│   ├── templates/            # Jinja2 页面模板
│   └── static/               # CSS / JS 静态资源
├── data/
│   ├── stopwords.txt         # 中文停用词表
│   └── sample_reviews.csv    # 开发用样本数据
├── output/                   # 生成的中间文件（gitignore）
└── tests/                    # 单元测试
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
python run.py

# 3. 打开浏览器访问 http://127.0.0.1:5000
```

首次使用时，通过 `/import` 页面上传 CSV 文件（格式见下文），系统会自动运行全流程处理。

## CSV 数据格式

```csv
review_id,user_id,dish_id,dish_name,restaurant,review_text,rating,date
1,U001,D001,宫保鸡丁,川味轩,鸡肉很嫩花生很脆麻辣味刚好,5,2026-01-15
```

| 字段 | 说明 |
|---|---|
| review_id | 评论 ID |
| user_id | 用户 ID |
| dish_id | 菜品 ID |
| dish_name | 菜品名称 |
| restaurant | 餐厅名称 |
| review_text | 评论文本 |
| rating | 评分（1-5） |
| date | 日期 |

## 页面路由

| 路由 | 说明 |
|---|---|
| `/` | 首页：用户列表 |
| `/user/<user_id>` | 用户画像：偏好标签 + 情感色彩 + 评论历史 |
| `/dish/<dish_id>` | 菜品画像：口味标签 + 频次 |
| `/recommend/<user_id>` | 推荐结果：Top-15 按相似度排名 |
| `/import` | CSV 上传页 |

## 配置

编辑 `config.py` 调整 NLP 参数：

```python
KEYWORD_TOP_K = 10        # 每条评论提取的关键词数量
KEYWORD_METHOD = 'tfidf'  # 关键词提取算法：'tfidf' 或 'textrank'
SENTIMENT_FLOOR = 0.3     # 负面关键词的最低权重系数
RECOMMEND_TOP_N = 15      # 推荐结果数量
```

## 运行测试

```bash
pytest tests/ -v
```
