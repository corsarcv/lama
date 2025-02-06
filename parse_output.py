import json
file_name = './data/output.dat'

batch_size = 5
all_stategies = []
with open(file_name, 'r') as f:
    print("Starting")
    while True:
        lines = [f.readline().strip() for _ in range(batch_size)]
        lines = [line for line in lines if line]
        print(lines)
        if not lines or len(lines) < batch_size:
             print('breaking')
             break

        print("*"*20)
        strategy = lines[2][lines[2].find('{'):]
        if 'Skipping startegy with pnl -' in lines[3]:
            print('Skipping loss strategy')
            continue
        account_stats = lines[4].replace('Account stats: ', '')
        strategy = strategy.replace("'", '"')
        strategy = strategy.replace('True', 'true')
        strategy = strategy.replace('False', 'false')
        account_stats = account_stats.replace("'", '"')
        print('Strategy:', strategy)
        print('Stats:', account_stats)
        
        account_stats = json.loads(account_stats)
        strategy = json.loads(strategy)

        all_stategies.append((strategy, account_stats))
all_stategies.sort(key = lambda e: e[1]['current_balance'], reverse=True)
for strategy, stat in all_stategies[:20]:
    print(strategy)
    print(stat)
    print("^"*30)



    