import requests
import json
# 使用中转链接可以和特定的API可以不必向openai发起请求，且请求无须魔法
# 调用方式与openai官网一致，仅需修改baseurl
Baseurl = "https://api.claudeshop.top"
Skey = "sk-nNyMfWpt7kwWhLaFBdDc5a1eEbF441239bB03d846d84E4Ea"
payload = json.dumps({
   "model": "gpt-4o-mini",
   "messages": [
      {
         "role": "system",
         "content": "You are a helpful assistant."
      },
      {
         "role": "user",
         "content": "hello"
      }
   ]
})
url = Baseurl + "/v1/chat/completions"
headers = {
   'Accept': 'application/json',
   'Authorization': f'Bearer {Skey}',
   'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

# 解析 JSON 数据为 Python 字典
data = response.json()

# 获取 content 字段的值
content = data

print(content)
