from typing import List, Dict
from math import log2


class IndexNode:
    def __init__(self, bloom_filter: str):
        self.bloom_filter = bloom_filter
        self.membership_value: float = 0.0
        self.inner_links: List[int] = []
        self.state = False
        self.external_links = {}
        self.indexes = (0, 0)

    def get_bloomfilter(self):
        return self.bloom_filter

    def set_membership_value(self, membership_value: float):
        self.membership_value = membership_value

    def add_inner_link(self, pointer: int):
        self.inner_links.append(pointer)

    def get_inner_links(self) -> List[int]:
        return self.inner_links

    def add_external_link(self, doc_path: str, mem_value: float):
        self.external_links[doc_path] = mem_value

    def get_external_links(self):
        return self.external_links

    def set_state(self, state: bool):
        self.state = state

    def get_state(self):
        return self.state

    def set_indexes(self, level: int, position: int):
        self.indexes = (level, position)

    def get_indexes(self):
        return self.indexes


class IndexStructure:
    def __init__(self, height):
        self.height: int = height
        self.structure: List[List[IndexNode]] = []
        self.bfs_for_level: List[Dict[str, int]] = []

    def initialization(self):
        self.structure = [[] for _ in range(self.height + 1)]
        self.bfs_for_level = [{} for _ in range(self.height + 1)]

    def get_node(self, level: int, position: int) -> IndexNode:
        return self.structure[level][position]

    @staticmethod
    def generate_ancestors(bloom_filter: str) -> List[str]:
        ones_count = bloom_filter.count('1')

        if ones_count < 1:
            return []
        predecessors = []

        for i in range(len(bloom_filter)):
            if bloom_filter[i] == '1':
                nueva_cadena = bloom_filter[:i] + '0' + bloom_filter[i + 1:]
                predecessors.append(nueva_cadena)
        return predecessors

    def build_branch(self, current_node: IndexNode) -> None:
        current_node_level, current_node_position = current_node.get_indexes()
        bloom_filter = current_node.get_bloomfilter()

        if current_node_level == 1:
            return

        ancestors = self.generate_ancestors(bloom_filter)
        for ancestor in ancestors:
            level = current_node_level - 1

            if ancestor in self.bfs_for_level[level]:
                position = self.bfs_for_level[level][ancestor]
                cn_inner_links = self.structure[level][position].get_inner_links()
                if current_node_position in cn_inner_links:
                    continue
                self.structure[level][position].add_inner_link(current_node_position)
                continue
            else:
                ancestor_node = IndexNode(ancestor)
                position = len(self.structure[level])
                ancestor_node.set_indexes(level, position)
                ancestor_node.set_state(True)
                ancestor_node.add_inner_link(current_node_position)
                self.structure[level].append(ancestor_node)
                self.bfs_for_level[level][ancestor] = position
                self.build_branch(ancestor_node)

    def insert_doc(self, doc_path: str, bfs_mvs: Dict[str, float]) -> None:
        for bloom_filter in list(bfs_mvs.keys()):
            level = bloom_filter.count('1')
            if bloom_filter in self.bfs_for_level[level]:
                position = self.bfs_for_level[level][bloom_filter]
                if doc_path in self.structure[level][position].get_external_links():
                    continue
                self.structure[level][position].add_external_link(doc_path, bfs_mvs[bloom_filter])
                continue

            new_node = IndexNode(bloom_filter)
            position = len(self.structure[level])
            new_node.set_state(True)
            new_node.set_indexes(level, position)
            new_node.add_external_link(doc_path, bfs_mvs[bloom_filter])
            self.structure[level].append(new_node)
            self.bfs_for_level[level][bloom_filter] = position
            self.build_branch(new_node)

    def docs_by_bloom_filters_search(self, bloom_filters: List[str]) -> List[str]:
        if len(bloom_filters) == 0:
            return []
        candidate_positions_for_level: List[List[int]] = [[] for _ in range(self.height + 1)]
        for level in range(1, self.height + 1):
            for bloom_filter in self.bfs_for_level[level]:
                if bloom_filter in bloom_filters:
                    position = self.bfs_for_level[level][bloom_filter]
                    if position in candidate_positions_for_level[level]:
                        continue
                    candidate_positions_for_level[level].append(position)
                    bloom_filters = [bf for bf in bloom_filters if bf != bloom_filter]
            if level > 1:
                for ancestor_position in candidate_positions_for_level[level - 1]:
                    ancestor_inner_links = self.structure[level - 1][ancestor_position].get_inner_links()
                    for inner_link in ancestor_inner_links:
                        if inner_link in candidate_positions_for_level[level]:
                            continue
                        candidate_positions_for_level[level].append(inner_link)

        docs_found: List[str] = []
        for level in range(1, self.height + 1):
            for position in candidate_positions_for_level[level]:
                docs = self.structure[level][position].get_external_links()
                docs_found.extend(docs)
        auxiliar_list = []
        for doc in docs_found:
            if doc not in auxiliar_list:
                auxiliar_list.append(doc)
        docs_found = auxiliar_list

        return docs_found

    def docs_extract_from_branch(self, level: int,
                                 position: int,
                                 qbf_index: int,
                                 docs_and_mv_for_ranking: List[Dict[str, List[float]]]
                                 ) -> None:

        external_links = self.structure[level][position].get_external_links()
        for external_link, mem_value in external_links.items():
            if external_link not in docs_and_mv_for_ranking[qbf_index]:
                docs_and_mv_for_ranking[qbf_index][external_link] = []
            docs_and_mv_for_ranking[qbf_index][external_link].append(mem_value)

        if level == self.height + 1:
            return

        inner_links = self.structure[level][position].get_inner_links()
        for position in inner_links:
            self.docs_extract_from_branch(level + 1, position, qbf_index, docs_and_mv_for_ranking)

    def docs_retrieved_from_qbfs_search(self,
                                        query_bfs: List[str],
                                        ) -> List[Dict[str, List[float]]]:
        if len(query_bfs) == 0:
            return []
        docs_and_mv_for_ranking = [{} for _ in range(len(query_bfs))]

        for qbf_index in range(len(query_bfs)):
            qbf = query_bfs[qbf_index]
            level = qbf.count('1')
            if qbf in self.bfs_for_level[level]:
                position = self.bfs_for_level[level][qbf]
                self.docs_extract_from_branch(level, position, qbf_index, docs_and_mv_for_ranking)
        return docs_and_mv_for_ranking

    def retrieved_docs_sorted(self,
                              query_bfs: List[str],
                              weights: List[float],
                              inclusion_relations: List[List[str]],
                              alpha: float = 0.25,
                              beta: float = 0.75,
                              ) -> List[str]:
        docs_and_mv_for_ranking = self.docs_retrieved_from_qbfs_search(query_bfs)
        docs = []
        docs_with_scores = {}
        for dictionary in docs_and_mv_for_ranking:
            docs.extend(dictionary.keys())
        docs = list(set(docs))

        for doc in docs:
            score = 0.0

            for qbf_index in range(len(docs_and_mv_for_ranking)):
                # This part is for the first part of the score function
                if doc in docs_and_mv_for_ranking[qbf_index]:
                    for mem_value in docs_and_mv_for_ranking[qbf_index][doc]:
                        score += (alpha * weights[qbf_index] * mem_value)

                # This part is for the second part of the score function
                divisor_sum = 0.0
                for partial_word in inclusion_relations[qbf_index]:
                    partial_word_index = query_bfs.index(partial_word)
                    membership_value = 0.0
                    if doc in docs_and_mv_for_ranking[partial_word_index]:
                        for mem_value in docs_and_mv_for_ranking[partial_word_index][doc]:
                            membership_value += mem_value
                        try:
                            membership_value /= len(docs_and_mv_for_ranking[partial_word_index][doc])
                        except ZeroDivisionError:
                            membership_value = 0.0
                    divisor_sum += membership_value
                if divisor_sum == 0.0:
                    qbf_entropy = 0.0
                else:
                    exterior_sum = 0.0
                    for partial_word in inclusion_relations[qbf_index]:
                        partial_word_index = query_bfs.index(partial_word)
                        membership_value = 0.0
                        if doc in docs_and_mv_for_ranking[partial_word_index]:
                            for mem_value in docs_and_mv_for_ranking[partial_word_index][doc]:
                                membership_value += mem_value
                            try:
                                membership_value /= len(docs_and_mv_for_ranking[partial_word_index][doc])
                            except ZeroDivisionError:
                                membership_value = 0.0
                        exterior_sum += log2(membership_value / divisor_sum)
                    qbf_entropy = (0.0 - exterior_sum)
                score -= (beta * qbf_entropy)

                docs_with_scores[doc] = score
        print(docs_with_scores)
        docs_sorted = dict(sorted(docs_with_scores.items(), key=lambda item: item[1], reverse=True))
        docs_sorted_list = list(docs_sorted.keys())
        return docs_sorted_list

# index = IndexStructure(5)
# index.initialization()
#
# index.insert_doc('path/to/doc1', {'10101': 0.5, '11110': 0.4})
# # index.insert_doc('path/to/doc2', {'01010': 0.6})
#
# for k in range(height):
#     for node in index.structure[k]:
#         print(k, node.get_bloomfilter(), node.get_inner_links(), node.get_external_links())
# print(index.bfs_for_level)
# # print(index.structure[1][2].get_bloomfilter(), ":", index.structure[1][2].get_external_links())
# #
# print(index.docs_by_bloom_filters_search(['10001', '00010']))
# print(index.docs_by_bloom_filters_search(['01000']))
# print(index.docs_by_bloom_filters_search(['00100']))
