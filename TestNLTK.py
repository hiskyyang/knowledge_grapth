import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
import networkx as nx
import matplotlib.pyplot as plt


class KnowledgeGraphBuilder:
    def __init__(self):
        pass

    def extract_nouns_verbs(self, text):
        sentences = sent_tokenize(text)
        nouns = set()
        verbs = set()
        for sentence in sentences:
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            for word, pos in pos_tags:
                word_lower = word.lower()
                if pos.startswith("NN"):
                    nouns.add(word_lower)
                elif pos.startswith("VB"):
                    verbs.add(word_lower)
        return sorted(list(nouns)), sorted(list(verbs))

    def extract_subject_predicate_object_triplets(self, text):
        sentences = sent_tokenize(text)
        triplets = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)

            subject = None
            predicate = None
            objects = []

            for i, (word, pos) in enumerate(pos_tags):
                word_lower = word.lower()
                if pos.startswith("NN") and subject is None:
                    subject = word_lower
                elif pos.startswith("VB"):
                    predicate = word_lower
                    for j in range(i + 1, len(pos_tags)):
                        obj_word, obj_pos = pos_tags[j]
                        if obj_pos.startswith("NN"):
                            objects.append(obj_word.lower())
                    break
            if subject and predicate and objects:
                for obj in objects:
                    triplets.append((subject, predicate, obj))
        return triplets

    def build_graph(self, triplets):
        graph = nx.DiGraph()
        for subject, predicate, obj in triplets:
            graph.add_node(subject)
            graph.add_node(obj)
            graph.add_edge(subject, obj, relation=predicate)
        return graph

    def visualize_graph(self, graph):
        pos = nx.spring_layout(graph, k=0.8, iterations=50)
        edge_labels = nx.get_edge_attributes(graph, "relation")

        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
            width=1,
            edge_color="gray",
        )
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5, alpha=0.7
        )

        plt.title("Knowledge Graph")
        plt.show()


def print_entities_relationships(builder, text):
    entities, relationships = builder.extract_nouns_verbs(text)
    print("Entities:", entities)
    print("Relationships:", relationships)


def generate_and_draw_knowledge_graph(builder, text):
    triplets = builder.extract_subject_predicate_object_triplets(text)
    print("\nExtracted Triplets (Subject, Predicate, Object):")
    for triplet in triplets:
        print(triplet)
    knowledge_graph = builder.build_graph(triplets)
    builder.visualize_graph(knowledge_graph)


def main():
    builder = KnowledgeGraphBuilder()
    with open("text.txt", "r", encoding="utf-8") as file:
        text = file.read()

    print_entities_relationships(builder, text)
    generate_and_draw_knowledge_graph(builder, text)


if __name__ == "__main__":
    main()
