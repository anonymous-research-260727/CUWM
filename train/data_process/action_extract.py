import re
if __name__ == "__main__":
    roots = ["data/train/data/word/bing_search/success", "data/train/data/word/m365/success", \
            "data/train/data/word/qabench/success", "data/train/data/word/wikihow/success", \
            "data/train/data/excel/bing_search/success", "data/train/data/excel/m365/success", \
            "data/train/data/excel/qabench/success", \
            "data/train/data/ppt/bing_search/success", "data/train/data/ppt/m365/success", \
            "data/train/data/ppt/qabench/success"]
    
    # roots = ["data/test/data/word/bing_search/success", "data/test/data/word/m365/success", \
    #         "data/test/data/word/qabench/success", "data/test/data/word/wikihow/success", \
    #         "data/test/data/excel/bing_search/success", "data/test/data/excel/m365/success", \
    #         "data/test/data/excel/qabench/success", \
    #         "data/test/data/ppt/bing_search/success", "data/test/data/ppt/m365/success", \
    #         "data/test/data/ppt/qabench/success"]
    for root in roots:

        full_path = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.endswith(".jsonl"):
                    full_path.append(os.path.join(dirpath, fname).replace('\\', '/'))
        for file_path in tqdm(full_path, desc="Processing files", total=len(full_path)):
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            for i in range(len(data)):
                # import pdb;pdb.set_trace()
                action = {"control_name": data[i]["step"]["action"]["control_test"], \
                          "function": data[i]["step"]["action"]["function"], \
                          "args": data[i]["step"]["action"]["args"]}
                name_idx = data[i]["step"]["screenshot_clean"].split('/')[-1]

                action = json.dumps(action)
                
                parts = root.split("/")
                parts[2] = 'image'   # 替换第二个 'data'
                new_root = "/".join(parts)

                outpath = new_root + '/' + file_path.split("/")[-1].split(".")[0] + '/' + name_idx.replace('png', 'txt')
                # if os.path.exists(outpath):
                #     continue
                # import pdb;pdb.set_trace()
                with open(outpath, "w", encoding="utf-8") as f:
                    f.write(action)