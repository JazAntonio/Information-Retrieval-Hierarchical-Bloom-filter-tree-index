import os
import sys
from typing import List
from Data_Access_Layer import IndexStructure


class Service:
    def __init__(self, index_height: int):
        self.index_structure = IndexStructure(index_height)

    def insert_document(self) -> None:
        while True:
            doc_path = input("Please enter the path to the file: ")
            if os.path.isfile(doc_path):
                break
            else:
                print("The file does not exist. Please try again.")

        while True:
            bfs_file_path = input("Enter the file path of the file that contains"
                                  " the Bloom filters for the document:")
            if os.path.isfile(bfs_file_path):
                break
            else:
                print("The file does not exist. Please try again.")

        bfs: List[str] = []
        mem_values: List[float] = []

        try:
            with open(bfs_file_path, 'r') as archivo:
                for linea in archivo:
                    if ':' in linea:
                        cadena, valor = linea.split(':')
                        bfs.append(cadena.strip())
                        mem_values.append(float(valor.strip()))
        except FileNotFoundError:
            print(f"El archivo {bfs_file_path} no fue encontrado.")
        except ValueError:
            print("Error al convertir el valor a flotante.")
        except Exception as e:
            print(f"Ocurrió un error: {e}")

        bfs_mvs = dict(zip(bfs, mem_values))
        self.index_structure.insert_doc(doc_path, bfs_mvs)
        print("Document uploaded successfully.")

        # for k in range(self.index_structure.height):
        #     for node in self.index_structure.structure[k]:
        #         print(k, node.get_bloomfilter(), node.get_inner_links(), node.get_external_links())

    def query_search(self) -> None:

        while True:
            query_data = input("Enter the file path of the query data: ")
            if os.path.isfile(query_data):
                break
            else:
                print("The file does not exist. Please try again.")

        qbfs = []
        weights = []
        inclusion_relations = []

        with open(query_data, 'r') as archivo:
            lines = archivo.readlines()

            for i in range(0, len(lines), 2):
                bf, weight = lines[i].strip().split(':')
                qbfs.append(bf)
                weights.append(float(weight))

                inclusion_relation = lines[i + 1].strip().split(',')
                inclusion_relations.append(inclusion_relation)

        relevant_docs = self.index_structure.retrieved_docs_sorted(qbfs, weights, inclusion_relations)

        if len(relevant_docs) == 0:
            print("No documents found.")
        else:
            print("The relevant documents are:")
            for index, doc in enumerate(relevant_docs):
                print(f"{index + 1}.- {doc}")

    # def delete_document(self):
    #     document_name = input("Enter the name of the document: ")
    #     result = self.index_structure.delete_document(document_name)
    #     if result:
    #         print("The document has been deleted.")
    #     else:
    #         print("The document does not found.")
