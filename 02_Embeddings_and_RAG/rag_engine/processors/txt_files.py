import os

class DigestTxtFile:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents

    def extract_metadata(self) -> dict:
        metadata = {
            "file_path": self.path,
            "file_type": "txt",
            "encoding": self.encoding,
            "total_documents": len(self.documents),
            "total_characters": sum(len(doc) for doc in self.documents)
        }
        
        if os.path.isfile(self.path):
            stat = os.stat(self.path)
            metadata.update({
                "file_size": stat.st_size,
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime
            })
        
        return metadata