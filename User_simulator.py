import re
import os
import random
import hashlib
import unicodedata
from math import exp
from bitarray import bitarray
from collections import defaultdict
from typing import List, Dict, Tuple
# from nltk import download
# download('punkt')  # Para tokenización
# download('stopwords')  # Para stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words('english'))


class BloomFilter:
    def __init__(self, filter_size: int, number_of_hashes: int):
        """
        Initializes a Bloom Filter with a specified size and number of hash functions.

        Parameters:
        ----------
        filter_size : int
            The size of the bit array for the Bloom Filter.
        number_of_hashes : int
            The number of hash functions to use (maximum of 8).

        Raises:
        ------
        ValueError
            If the number of hash functions exceeds 8.
        """
        self.size = filter_size
        self.num_hashes = number_of_hashes

        if number_of_hashes > 8:
            raise ValueError("There must be a maximum of 8 hash functions.")

        self.bit_array = bitarray(filter_size)
        self.bit_array.setall(0)

        self.hash_functions = [
            hashlib.sha256, hashlib.sha384, hashlib.sha512,
            hashlib.sha3_224, hashlib.sha3_256, hashlib.sha3_384,
            hashlib.sha3_512, hashlib.sha224
        ]

    def get_cardinality(self) -> int:
        """
        Returns the count of set bits in the Bloom Filter.

        Returns:
        -------
        int
            The number of bits set to 1 in the bit array.
        """
        return self.bit_array.count()

    def get_bit_array(self) -> bitarray:
        """
        Returns the bit array of the Bloom Filter.

        Returns:
        -------
        bitarray
            The internal bit array representing the Bloom Filter.
        """
        return self.bit_array

    def _indices(self, item: str) -> List[int]:
        """
        Computes the indices in the bit array for a given item using the hash functions.

        Parameters:
        ----------
        item : str
            The item for which to compute the hash indices.

        Returns:
        -------
        List[int]
            A list of indices corresponding to the hash values in the bit array.
        """
        indices = []
        for i in range(self.num_hashes):
            hash_func = self.hash_functions[i]
            result = int(hash_func(item.encode()).hexdigest(), 16)  # Generates the hash
            indices.append(result % self.size)  # Index within the desired range
        return indices

    def add(self, item: str) -> None:
        """
        Adds an item to the Bloom Filter by setting the corresponding bits.

        Parameters:
        ----------
        item : str
            The item to be added to the Bloom Filter.

        Returns:
        -------
        None
        """
        for index in self._indices(item):
            self.bit_array[index] = 1

    def __contains__(self, item: str) -> bool:
        """
        Checks if an item is in the Bloom Filter.

        Parameters:
        ----------
        item : str
            The item to check for membership in the Bloom Filter.

        Returns:
        -------
        bool
            True if the item is possibly in the Bloom Filter, False if it is definitely not.
        """
        return all(self.bit_array[index] for index in self._indices(item))


class User:
    """
       Class that represents a user in the system.

       Attributes:
           name (str): The name of the user.
           email (str): The email address generated from the user's name.

       Methods:
           get_name: Returns the name of the user.
           get_email: Returns the user's email address.
           preprocess_doc: Preprocesses the document at the given path.
           generate_cw_with_frequencies: Generates a list of keywords and their frequencies from a text.
           calculate_membership_values: Calculates the membership values of the keywords.
           generate_bfs_with_mem_values: Generates Bloom filters and membership values from a document.
           generate_file_with_bfs_and_mvs: Generates a file that maps Bloom filters to their membership values.
    """

    def __init__(self, name):
        self.name = name
        self.email = "{0}@example.com".format(name)

    def get_name(self) -> str: return self.name

    def get_email(self) -> str: return self.email

    @staticmethod
    def preprocess_doc(document_path: str) -> str:
        """
        Preprocesses a document by reading its content, removing non-word characters and digits.

        Parameters:
        document_path (str): The path to the document to be processed.

        Returns:
        str: The processed text with non-word characters, accents, and digits removed.

        Raises:
        Exception: If there is an error reading the file, it prints the error message.
        """
        try:
            with open(document_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""  # Return an empty string if there's an error

        # Normalizes the text to its decomposed form
        text = unicodedata.normalize('NFD', text)
        # Filters out characters that are not accents (combining diacritical marks)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Remove non-word characters and digits
        text = re.sub("[^a-zA-Z ]", "", text)

        return text

    @staticmethod
    def generate_cw_with_frequencies(text: str, max_len_of_a_cw: int) -> Dict[str, int]:
        """
        Generates a dictionary of compound words and their frequencies from the given text.

        This static method tokenizes the input text, filters out stop words, and counts the
        occurrences of single words and compound words formed by consecutive non-stop words
        up to a specified maximum length.

        Parameters:
        ----------
        text : str
            The input text from which to generate compound words and their frequencies.
        max_len_of_a_cw : int
            The maximum length of the compound words to consider.

        Returns:
        -------
        Dict[str, int]
            A dictionary where the keys are the compound words (including single words)
            and the values are their respective frequencies in the input text.

        Notes:
        -----
        - The function uses the `word_tokenize` function from the NLTK library to tokenize
          the input text.
        - Stop words are filtered out, meaning they will not be counted towards the
          compound words.
        - The method constructs compound words by combining consecutive non-stop words,
          with a length limit defined by `max_len_of_a_cw`.

        Example:
        --------
        Code:
            frequencies = User.generate_cw_with_frequencies("Hello world", 2)
            print(frequencies)
        Output:
            {'Hello': 1, 'world': 1, 'Hello world': 1}
        """

        all_words = word_tokenize(text)
        compound_words = defaultdict(int)

        for i in range(len(all_words)):
            current_word = all_words[i]
            if current_word in english_stopwords or current_word == "":
                continue
            compound_words[current_word] += 1

            compound_word = current_word
            for j in range(1, max_len_of_a_cw):
                if i + j >= len(all_words) or all_words[i + j] in english_stopwords:
                    break
                compound_word = f"{compound_word} {all_words[i + j]}".strip()
                compound_words[compound_word] += 1

        return dict(compound_words)

    @staticmethod
    def calculate_membership_values(cw_with_frequencies: Dict[str, int]) -> Dict[str, float]:
        return {key: value / len(cw_with_frequencies) for key, value in cw_with_frequencies.items()}

    def generate_bfs_with_mvs_for_docs(
            self,
            doc_path: str,
            bfs_width: int,
            num_hashes: int,
            max_len_of_a_cw: int,
            max_cw_for_a_filter: int,
            theta: float
    ) -> Tuple[List[str], List[float], str]:
        """
        Generates Bloom Filters and their average membership values from the document at the given path.

        This method preprocesses the document, generates compound words with their frequencies,
        calculates membership values, and creates Bloom filters (for clusters of words that have
        nearby membership values), with a membership value calculated as the average of the membership
        values of the words they contain.

        Parameters:
        ----------
        doc_path : str
            The path to the document that needs to be processed.
        bfs_width : int
            The width of the Bloom Filter (size of the bit array).
        num_hashes : int
            The number of hash functions to use in the Bloom Filter.
        max_len_of_a_cw : int
            The maximum length of the compound words to consider.
        max_cw_for_a_filter : int
            The maximum number of compound words to include in a single Bloom Filter.
        theta : float
            The threshold for considering nearby membership values.

        Returns:
        -------
        Tuple[List[str], List[float]]
            A tuple containing:
            - A list of Bloom Filters generated.
            - A list of average membership values corresponding to each Bloom Filter.
            - The path of the document that was processed.

        Notes:
        -----
        - This method utilizes other methods such as `preprocess_doc`,
          `generate_cw_with_frequencies`, and `calculate_membership_values`.
        - It constructs Bloom Filters based on membership values that are within the specified threshold (theta).

        Example:
        --------
        bloom_filters, avg_membership_values = User.generate_bfs_with_mem_values("path/to/doc.txt", 100, 5, 10, 0.1)
        """

        preprocessed_doc = self.preprocess_doc(doc_path)
        cw_with_frequencies = self.generate_cw_with_frequencies(preprocessed_doc, max_len_of_a_cw)
        cw_with_membership_values = self.calculate_membership_values(cw_with_frequencies)

        len_of_cw_dic = len(cw_with_membership_values)
        if len_of_cw_dic == 0:
            return [], [], doc_path

        bloom_filters = []
        average_membership_values = []

        sorted_dict = dict(sorted(cw_with_membership_values.items(), key=lambda item: item[1], reverse=True))
        keys_listed = list(sorted_dict.keys())

        i = 0
        while len_of_cw_dic != 0:
            average_mem = 0
            t = sorted_dict.get(keys_listed[i])
            nearby_words = {cw: mem for cw, mem in sorted_dict.items() if abs(mem - t) <= theta}

            bf = BloomFilter(bfs_width, num_hashes)

            if len(nearby_words) <= max_cw_for_a_filter:
                for word_key in nearby_words.keys():
                    bf.add(word_key)
                for word_value in nearby_words.values():
                    average_mem += word_value
                i = i + len(nearby_words)
                for word_key in nearby_words:
                    del sorted_dict[word_key]
                try:
                    average_mem /= len(nearby_words)
                except ZeroDivisionError:
                    average_mem = 0
            else:
                rand_num = random.randint(0, min(len_of_cw_dic, max_cw_for_a_filter))
                for k in range(rand_num):
                    bf.add(keys_listed[i + k])
                    average_mem += sorted_dict[keys_listed[i + k]]
                    del sorted_dict[keys_listed[i + k]]
                i = i + rand_num
                try:
                    average_mem /= rand_num
                except ZeroDivisionError:
                    average_mem = 0

            if bf.get_cardinality() != 0:
                bloom_filters.append(bf.get_bit_array().to01())
                average_membership_values.append(average_mem)

            len_of_cw_dic = len(sorted_dict)

        return bloom_filters, average_membership_values, doc_path

    @staticmethod
    def generate_file_to_upload_for_doc(
            bfs: List[str],
            mem_values: List[float],
            doc_path: str
    ) -> None:
        """
        Generates a file that maps Bloom Filters (BFs) to their corresponding
        membership values (MVs).

        Args:
            bfs (List[str]): A list of Bloom Filter names (as strings).
            mem_values (List[float]): A list of corresponding membership values (as floats).
            doc_path (str): The path to the document where the output file
                will be saved. The output file will be named based on this
                path with '_bfs' appended to the filename.

        Returns:
            None: This function does not return any value. It writes the
                output to a file.

        Raises:
            ValueError: If the lengths of the two lists do not match, a message
            will be printed indicating that the lists must have the same length.

        The generated file will contain lines formatted as follows:
        "BloomFilter : AverageMembershipValue", one for each Bloom Filter
        and its corresponding membership value.
        """
        # Ensure both lists have the same length
        if len(bfs) != len(mem_values):
            print("The lists must have the same length.")
            return

        # Create a file name based on the document path
        base, ext = os.path.splitext(doc_path)
        # Create the new file path
        nueva_ruta = f"{base}_bfs{ext}"

        with open(nueva_ruta, 'w') as file:
            # Write each Bloom Filter and its corresponding membership value
            for bf, mem_value in zip(bfs, mem_values):
                file.write(f"{bf} : {mem_value}\n")

    @staticmethod
    def generate_qcw_with_weights(original_query_tokens: List[str],
                                  original_weights: List[float],
                                  max_len_query_cw: int) -> Tuple[List[str], List[float]]:

        original_qcw_and_weights: Dict[str, int] = dict(zip(original_query_tokens, original_weights))

        query_cw: List[str] = []
        for i in range(len(original_query_tokens)):
            token = original_query_tokens[i]
            if token in english_stopwords or token == "":
                continue
            query_cw.append(token)

            compound_token = token
            for j in range(1, max_len_query_cw):
                if ((i + j) >= len(original_query_tokens)) or (original_query_tokens[i + j] in english_stopwords):
                    break
                compound_token = f"{compound_token} {original_query_tokens[i + j]}".strip()
                query_cw.append(compound_token)
        query_cw = list(dict.fromkeys(query_cw))

        weights: List[float] = [0.0] * len(query_cw)
        for i in range(len(query_cw)):
            if query_cw[i] in original_query_tokens:
                weights[i] = original_qcw_and_weights[query_cw[i]]
            else:
                total_sum = 0.0
                tokens_of_a_qcw = word_tokenize(query_cw[i])
                for compound_token in tokens_of_a_qcw:
                    quotient = len(original_query_tokens) / len(tokens_of_a_qcw)
                    partial_weight = original_qcw_and_weights[compound_token] * exp(quotient - 1)
                    total_sum += partial_weight
                weights[i] = total_sum

        return query_cw, weights

    def generate_qbfs_with_ws_and_inclusion_relations(self,
                                                      query: str,
                                                      bf_length: int,
                                                      num_hashes: int,
                                                      max_len_query_cw: int) \
            -> Tuple[List[str], List[float], List[List[str]]]:
        weights = []
        while True:
            try:
                option_for_weights = int(input("Do you want to assign weights to the query words?: "
                                               "\n1 := Yes\n2 := No\n"))
                break
            except ValueError:
                print("Invalid input, try again.")

        if option_for_weights == 1:
            print("Assign values, only integers values:")
            query_tokens = word_tokenize(query)
            query_tokens = list(dict.fromkeys(query_tokens))
            for word in query_tokens:
                while True:
                    try:
                        weight = float(input(f"Enter the weight for the word \"{word}\": "))
                        break
                    except ValueError:
                        print("Invalid input, try again.")
                weights.append(weight)
        else:
            query_tokens = word_tokenize(query)
            query_tokens = list(dict.fromkeys(query_tokens))
            weights = [1.0] * len(query_tokens)

        qcws, weights = self.generate_qcw_with_weights(query_tokens, weights, max_len_query_cw)
        qcws_and_qbfs = {}
        for i in range(len(qcws)):
            bf = BloomFilter(bf_length, num_hashes)
            bf.add(qcws[i])
            qbf = bf.get_bit_array().to01()
            qcws_and_qbfs[qcws[i]] = qbf

        inclusion_relations = []

        for qcw in qcws:
            bf_list_for_qcw = []
            words = word_tokenize(qcw)
            for word in words:
                bf_list_for_qcw.append(qcws_and_qbfs[word])
            inclusion_relations.append(bf_list_for_qcw)

            # if len(words) == 1:
            #     inclusion_relations.append([])
            # else:
            #     for word in words:
            #         bf_list_for_qcw.append(qcws_and_qbfs[word])
            #     inclusion_relations.append(bf_list_for_qcw)

        qbfs = list(qcws_and_qbfs.values())

        return qbfs, weights, inclusion_relations

    @staticmethod
    def generate_file_for_qbfs(path_to_store_file: str,
                               bfs_of_qcw: List[str],
                               qcw_weights: List[float],
                               inclusion_relations: List[List[str]]) -> None:
        # Create a file name based on the document path
        base, ext = os.path.splitext(path_to_store_file)
        # Create the new file path
        nueva_ruta = f"{base}_qbfs{ext}"
        with open(nueva_ruta, 'w') as file:
            for i in range(len(bfs_of_qcw)):
                file.write(f"{bfs_of_qcw[i]}:{qcw_weights[i]}\n")
                file.write(",".join(inclusion_relations[i]) + "\n")


def main():
    data_owner = User("proprietary")
    doc_path = "./Carpeta_origen/Pruebas/cuento2.txt"
    qbfs_path = "./Carpeta_origen/Pruebas/second_query.txt"
    bloom_filter_len = 28
    num_hashes = 3
    max_len_cw = 3
    max_words_for_bf = 3
    theta = 0.01
    max_len_qcw = 3

    # doc_path = input("Enter the path to the document: ")
    bfs, ave_memv, doc_path = data_owner.generate_bfs_with_mvs_for_docs(doc_path, bloom_filter_len, num_hashes, max_len_cw, max_words_for_bf, theta)
    data_owner.generate_file_to_upload_for_doc(bfs, ave_memv, doc_path)
    # query = input("Enter the query you want to search: ")
    # qbfs, w, inclusion = data_owner.generate_qbfs_with_ws_and_inclusion_relations(query, bloom_filter_len, num_hashes, max_len_qcw)
    # data_owner.generate_file_for_qbfs(qbfs_path, qbfs, w, inclusion)


if __name__ == "__main__":
    main()
