import spacy
import networkx as nx
import matplotlib.pyplot as plt
import os


class KnowledgeGraphBuilderSpaCy:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Downloading...")
            from spacy.cli import download

            download(model_name)
            self.nlp = spacy.load(model_name)

    def extract_entities_relationships(self, text):
        doc = self.nlp(text)
        nouns = set()
        verbs = set()

        for sent in doc.sents:
            for chunk in sent.noun_chunks:
                nouns.add(chunk.text.strip())

            for token in sent:
                if token.pos_ in ["VERB", "AUX"]:
                    verbs.add(token.lemma_)

        return list(sorted(nouns)), list(sorted(verbs))

    def extract_triplets(self, text):
        doc = self.nlp(text)
        triplets = []

        for sent in doc.sents:
            noun_chunks = {chunk.root: chunk.text for chunk in sent.noun_chunks}
            subject = None

            for token in sent:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    subject = noun_chunks.get(token, token.text)

                if token.pos_ in ["VERB", "AUX"]:
                    relation = token.lemma_
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

                    if subject and obj:
                        triplets.append(
                            (subject.strip(), relation.strip(), obj.strip())
                        )

        print("Extracted Triplets:", triplets)
        return triplets

    def generate_knowledge_graph(self, triplets):
        graph = nx.DiGraph()

        for subject, relation, object_ in triplets:
            print(f"Adding to graph: {subject} -[{relation}]-> {object_}")
            graph.add_node(subject)
            graph.add_node(object_)
            graph.add_edge(subject, object_, relation=relation)

        print("Graph Nodes:", list(graph.nodes))
        print("Graph Edges:", list(graph.edges))

        return graph

    def visualize_graph(self, graph):
        pos = nx.spring_layout(graph, k=0.9, iterations=50)
        edge_labels = nx.get_edge_attributes(graph, "relation")

        plt.figure(figsize=(10, 7))
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
            graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5, alpha=0.8
        )

        plt.title("Knowledge Graph (spaCy)")
        plt.show()


def print_spacy_entities_relationships(builder, text):
    nouns, verbs = builder.extract_entities_relationships(text)
    print(f"Entities: {', '.join(sorted(nouns))}")
    print(f"Relationships(: {', '.join(sorted(verbs))}")


def generate_and_draw_spacy_knowledge_graph(builder, text):
    triplets = builder.extract_triplets(text)
    knowledge_graph = builder.generate_knowledge_graph(triplets)
    builder.visualize_graph(knowledge_graph)


def main():
    builder = KnowledgeGraphBuilderSpaCy()

    file_path = "text.txt"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    print_spacy_entities_relationships(builder, text)

    generate_and_draw_spacy_knowledge_graph(builder, text)


if __name__ == "__main__":
    main()
