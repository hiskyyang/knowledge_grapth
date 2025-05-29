import spacy
import networkx as nx
import matplotlib.pyplot as plt
import os


class KnowledgeGraphBuilderSpaCy:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Downloading...")
            from spacy.cli import download

            download(model_name)
            self.nlp = spacy.load(model_name)

    def _extract_nouns(self, doc):
        return sorted(
            list(
                set(
                    chunk.text.strip()
                    for sent in doc.sents
                    for chunk in sent.noun_chunks
                )
            )
        )

    def _extract_verbs(self, doc):
        return sorted(
            list(
                set(
                    token.text
                    for sent in doc.sents
                    for token in sent
                    if token.pos_ in ["VERB", "AUX"]
                )
            )
        )

    def extract_entities_relationships(self, text):
        doc = self.nlp(text)
        nouns = self._extract_nouns(doc)
        verbs = self._extract_verbs(doc)
        return nouns, verbs

    def _process_clause(self, token, noun_chunks, parent_entity=None):
        subject = parent_entity
        relation = None
        obj = None
        triplets = []

        if token.pos_ in ["VERB", "AUX"]:
            relation = token.text

            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = noun_chunks.get(child, child.text)
                elif child.dep_ in ["dobj", "pobj", "iobj", "attr", "prep", "obl"]:
                    obj = noun_chunks.get(child, child.text)
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ in ["pobj", "obl"]:
                                obj = noun_chunks.get(grandchild, grandchild.text)
                                break
                    if child.dep_ in ["acl", "relcl"]:
                        triplets.extend(
                            self._process_clause(child, noun_chunks, parent_entity=obj)
                        )

            if subject and relation and obj:
                triplets.append((subject.strip(), relation.strip(), obj.strip()))

        return triplets

    def extract_triplets(self, text):
        doc = self.nlp(text)
        all_triplets = []

        for sent in doc.sents:
            noun_chunks = {chunk.root: chunk.text for chunk in sent.noun_chunks}
            subject = None

            for token in sent:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    subject = noun_chunks.get(token, token.text)

                if token.pos_ in ["VERB", "AUX"]:
                    relation = token.text
                    obj = None

                    for child in token.children:
                        if child.dep_ in [
                            "dobj",
                            "pobj",
                            "iobj",
                            "attr",
                            "prep",
                            "obl",
                        ]:
                            obj = noun_chunks.get(child, child.text)
                            if child.dep_ == "prep":
                                for grandchild in child.children:
                                    if grandchild.dep_ in ["pobj", "obl"]:
                                        obj = noun_chunks.get(
                                            grandchild, grandchild.text
                                        )
                                        break
                            if child.dep_ in ["acl", "relcl"]:
                                all_triplets.extend(
                                    self._process_clause(
                                        child, noun_chunks, parent_entity=obj
                                    )
                                )
                            break

                    if subject and relation and obj:
                        all_triplets.append(
                            (subject.strip(), relation.strip(), obj.strip())
                        )
                    elif subject and relation and not list(token.children):
                        all_triplets.append((subject.strip(), relation.strip(), ""))

        print("Extracted Triplets:", all_triplets)
        return all_triplets

    def generate_knowledge_graph(self, triplets):
        graph = nx.DiGraph()
        for subject, relation, object_ in triplets:
            print(f"Adding to graph: {subject} -[{relation}]-> {object_}")
            graph.add_node(subject)
            if object_:
                graph.add_node(object_)
                graph.add_edge(subject, object_, relation=relation)
            else:
                graph.add_node(relation)
                graph.add_edge(subject, relation, relation="action")
        print("Graph Nodes:", list(graph.nodes))
        print("Graph Edges:", list(graph.edges))
        return graph

    def visualize_graph(self, graph):
        pos = nx.spring_layout(graph, k=0.9, iterations=50)
        edge_labels = nx.get_edge_attributes(graph, "relation")

        plt.figure(figsize=(12, 10))
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="lightblue",
            font_size=8,
            width=1.2,
            edge_color="gray",
            alpha=0.9,
        )
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5, alpha=0.8
        )

        plt.title("Knowledge Graph (spaCy)", fontsize=16)
        plt.show()


def print_spacy_entities_relationships(builder, text):
    entities, relationships = builder.extract_entities_relationships(text)
    print(f"Entities: {', '.join(entities)}")
    print(f"Relationships: {', '.join(relationships)}")


def generate_and_draw_spacy_knowledge_graph(builder, text):
    triplets = builder.extract_triplets(text)
    knowledge_graph = builder.generate_knowledge_graph(triplets)
    builder.visualize_graph(knowledge_graph)


def main():
    builder = KnowledgeGraphBuilderSpaCy()
    file_path = "text.txt"
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    print_spacy_entities_relationships(builder, text)
    generate_and_draw_spacy_knowledge_graph(builder, text)


if __name__ == "__main__":
    main()
