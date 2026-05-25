from huggingface_hub import list_repo_files

REPO_ID = "minwoosun/CholecSeg8k"

def main():
    files = list_repo_files(REPO_ID, repo_type="dataset")
    print("Here are files in the CholecSeg8k dataset repo :")
    for a_file in files:
        print(f"- {a_file}")


if __name__ == "__main__":
    main()
