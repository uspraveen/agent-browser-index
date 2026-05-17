import sys

with open('main.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    'MIMO_API_URL, MAIN_LLM_MODEL',
    'MIMO_API_URL, MIMO_API_KEY, MAIN_LLM_MODEL'
)

text = text.replace(
    'mimo_api_url: str = MIMO_API_URL,\n    headless: bool = False,',
    'mimo_api_url: str = MIMO_API_URL,\n    mimo_api_key: str = MIMO_API_KEY,\n    headless: bool = False,'
)

text = text.replace(
    'parser.add_argument("--mimo-url", type=str, default="http://cosmos-9.ddns.ualr.edu:8098", help="MiMo API URL.")',
    'parser.add_argument("--mimo-url", type=str, default=MIMO_API_URL, help="MiMo API URL.")\n    parser.add_argument("--mimo-api-key", type=str, default=MIMO_API_KEY, help="MiMo API Key for authentication.")'
)

text = text.replace(
    'check_mimo_health(args.mimo_url)',
    'check_mimo_health(args.mimo_url, api_key=args.mimo_api_key)'
)

text = text.replace(
    'mimo_api_url=args.mimo_url,\n            headless=args.headless,',
    'mimo_api_url=args.mimo_url,\n            mimo_api_key=args.mimo_api_key,\n            headless=args.headless,'
)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Patched main.py successfully.')
