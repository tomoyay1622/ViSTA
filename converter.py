import sys
import os

class SequentialCounter:
    """
    濃度制約（Cardinality Constraint）を順序回路（Sequential Counter）を用いて
    通常ルールへ変換するクラス。
    """
    def __init__(self, translator, literals, lower_bound):
        self.translator = translator
        self.literals = literals
        self.lower_bound = lower_bound
        self.n = len(literals)
        self.state_atoms = {}

    def get_state_id(self, i, j):
        """状態 S_{i, j} (i番目まで見てj個以上真) のIDを取得"""
        if j <= 0: return None # 常にTrue
        if j > i: return None # 常にFalse
        if (i, j) not in self.state_atoms:
            self.state_atoms[(i, j)] = self.translator.get_new_id()
        return self.state_atoms[(i, j)]

    def generate_circuit(self):
        """回路生成のメインロジック"""
        if self.lower_bound <= 0: return "TRUE"
        if self.lower_bound > self.n: return "FALSE"

        for i in range(1, self.n + 1):
            lit = self.literals[i-1]
            max_j = min(i, self.lower_bound)
            
            for j in range(1, max_j + 1):
                current_s = self.get_state_id(i, j)
                
                # 1. 継承 S_{i-1, j} -> S_{i, j}
                prev_s_same = self.get_state_id(i - 1, j)
                if prev_s_same is not None:
                    self.translator.emit_normal_rule(current_s, [prev_s_same])
                
                # 2. カウントアップ S_{i-1, j-1} & lit -> S_{i, j}
                prev_s_minus = self.get_state_id(i - 1, j - 1)
                if prev_s_minus is not None:
                    self.translator.emit_normal_rule(current_s, [prev_s_minus, lit])
                elif j == 1:
                    # j=1の場合、前の状態はカウント0(True)なので lit だけで成立
                    self.translator.emit_normal_rule(current_s, [lit])

        return self.get_state_id(self.n, self.lower_bound)


class AspifTranslator:
    def __init__(self):
        self.max_id = 0
        self.output_rules = []

    def get_new_id(self):
        self.max_id += 1
        return self.max_id

    def emit_normal_rule(self, head, body_literals):
        """通常ルールを出力形式(ASPIF準拠のNormal Rule)で保存"""
        head_list = [head] if (head is not None and head != 0) else []
        head_len = len(head_list)
        body_len = len(body_literals)
        # 1 (rule) 0 (head_type:normal) (HeadLen) (Head...) 0 (BodyLen) (Body...)
        parts = [1, 0, head_len] + head_list + [0, body_len] + body_literals
        self.output_rules.append(" ".join(map(str, parts)))

    def process_choice_rule(self, head_atoms, body_literals):
        """ 選択ルールの変換 """
        aux_body = self.get_new_id()
        self.emit_normal_rule(aux_body, body_literals)

        for h in head_atoms:
            aux_not_h = self.get_new_id()
            self.emit_normal_rule(h, [aux_body, -aux_not_h])
            self.emit_normal_rule(aux_not_h, [-h])

    def process_weight_rule(self, head_atom, lower_bound, literals):
        """ 重み付きルールの変換 """
        counter = SequentialCounter(self, literals, lower_bound)
        circuit_output = counter.generate_circuit()

        if circuit_output == "TRUE":
            self.emit_normal_rule(head_atom, [])
        elif circuit_output == "FALSE":
            pass
        else:
            target_head = head_atom if head_atom != 0 else 0
            self.emit_normal_rule(target_head, [circuit_output])

    def run(self, aspif_text):
        lines = aspif_text.strip().split('\n')
        
        # Pass 1: ID最大値のスキャン
        for line in lines:
            tokens = line.split()
            if not tokens: continue
            if tokens[0] != '1': continue # ルール行以外は無視
            
            parts = list(map(int, tokens))
            self.max_id = max(self.max_id, max(abs(x) for x in parts))

        # Pass 2: 変換処理
        for line in lines:
            tokens = line.split()
            if not tokens: continue

            # パススルー: ルール以外はそのまま出力
            if tokens[0] != '1':
                self.output_rules.append(line)
                continue

            parts = list(map(int, tokens))
            idx = 1
            
            # Head
            head_type = parts[idx]; idx += 1
            head_len = parts[idx]; idx += 1
            head_atoms = parts[idx : idx + head_len]
            idx += head_len
            
            # Body
            body_type = parts[idx]; idx += 1
            
            body_data = {}
            if body_type == 0: # Normal Body
                body_len = parts[idx]; idx += 1
                body_lits = parts[idx : idx + body_len]
                idx += body_len
                body_data = {'literals': body_lits}
                
            elif body_type == 1: # Weight Body
                lower_bound = parts[idx]; idx += 1
                body_len = parts[idx]; idx += 1
                lits = []
                for _ in range(body_len):
                    l = parts[idx]
                    lits.append(l)
                    idx += 2
                body_data = {'bound': lower_bound, 'literals': lits}

            # --- 分岐処理 ---
            
            # Case A: 選択ルール + 重み付きボディ
            if head_type == 1 and body_type == 1:
                aux_atom = self.get_new_id()
                self.process_weight_rule(aux_atom, body_data['bound'], body_data['literals'])
                self.process_choice_rule(head_atoms, [aux_atom])

            # Case B: 選択ルール
            elif head_type == 1:
                self.process_choice_rule(head_atoms, body_data['literals'])

            # Case C: 重み付きルール
            elif body_type == 1:
                h = head_atoms[0] if head_atoms else 0
                self.process_weight_rule(h, body_data['bound'], body_data['literals'])

            # Case D: 通常ルール
            else:
                h = head_atoms[0] if head_atoms else 0
                self.emit_normal_rule(h, body_data['literals'])

    def print_results(self):
        # 変換結果のみを標準出力へ（パイプライン処理のため余計な装飾は削除推奨）
        for r in self.output_rules:
            print(r)

if __name__ == '__main__':
    translator = AspifTranslator()
    
    # 入力の読み込み
    content = ""
    
    # 1. コマンドライン引数でファイルが指定された場合
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
        else:
            sys.stderr.write(f"Error: File not found: {filepath}\n")
            sys.exit(1)
            
    # 2. 引数がなく、標準入力にデータがある場合 (パイプライン)
    elif not sys.stdin.isatty():
        content = sys.stdin.read()
        
    else:
        # 入力がない場合
        sys.stderr.write("Usage: python asp_translator.py <file.aspif>\n")
        sys.stderr.write("   OR: cat file.aspif | python asp_translator.py\n")
        sys.exit(1)

    # 実行
    translator.run(content)
    translator.print_results()