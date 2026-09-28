"""Graphe SNOMED CT (RF2) pour synonymes et hiérarchie."""
import re
from itertools import groupby
from typing import Dict, Generator, List, Set, Tuple

import networkx as nx
import pandas as pd

from more_itertools import pairwise


class SnomedConceptDetails:
    """
    Détails essentiels d'un concept SNOMED CT.
    """

    def __init__(self, sctid: int, fsn: str, synonyms: List[str] = None) -> None:
        self.sctid = sctid
        self.fsn = fsn
        self.synonyms = synonyms

    def __repr__(self):
        return f"{self.sctid} | {self.fsn}"

    def __eq__(self, other):
        return self.sctid == other.sctid

    def __hash__(self):
        return self.sctid

    @property
    def hierarchy(self) -> str:
        hierarchy_match = re.search(r"\(([^)]+)\)\s*$", self.fsn)


class SnomedRelationship:
    """
    Relation SNOMED CT.
    """

    def __init__(
        self,
        src: SnomedConceptDetails,
        tgt: SnomedConceptDetails,
        group: int,
        type: str,
        type_id: str,
    ) -> None:
        self.src = src
        self.tgt = tgt
        self.group = group
        self.type = type
        self.type_id = type_id

    def __repr__(self):
        return f"[{self.src}] ---[{self.type}]---> [{self.tgt}]"


class SnomedRelationshipGroup:
    def __init__(self, group: int, relationships: List[SnomedRelationship]) -> None:
        self.group = group
        self.relationships = relationships

    def __repr__(self):
        return f"Group {self.group}\n\t" + "\n\t".join([str(r) for r in self.relationships])


class SnomedConcept:
    def __init__(self, concept_details, parents, children, inferred_relationship_groups) -> None:
        self.concept_details = concept_details
        self.inferred_relationship_groups = inferred_relationship_groups
        self.parents = parents
        self.children = children

    def __repr__(self):
        str_ = str(self.concept_details)
        str_ += f"\n\nSynonyms:\n{self.concept_details.synonyms}"
        str_ += "\n\nParents:\n"
        str_ += "\n".join([str(p) for p in self.parents])
        str_ += "\n\nChildren:\n"
        str_ += "\n".join([str(c) for c in self.children])
        str_ += "\n\nInferred Relationships:\n"
        str_ += "\n".join([str(rg) for rg in self.inferred_relationship_groups])
        return str_

    @property
    def sctid(self) -> int:
        return self.concept_details.sctid

    @property
    def fsn(self) -> str:
        return self.concept_details.fsn

    @property
    def synonyms(self) -> List[str]:
        return self.concept_details.synonyms

    @property
    def hierarchy(self) -> str:
        return self.concept_details.hierarchy


class SnomedGraph:
    """
    Release SNOMED CT représentée comme un graphe NetworkX.

    Attributs
    ---------
    G : nx.DiGraph
        Le graphe sous-jacent
    """

    fsn_typeId = 900000000000003001
    is_a_relationship_typeId = 116680003
    root_concept_id = 138875005

    def __init__(self, G: nx.DiGraph) -> None:
        """
        Crée une instance SnomedGraph à partir d'un DiGraph NetworkX

        Args :
            G: DiGraph issu de SnomedGraph.from_rf2() ou SnomedGraph.from_serialized().
        Retourne :
            self.
        """
        self.G = G
        print(self)

    def __repr__(self):
        return f"SNOMED graph has {self.G.number_of_nodes()} vertices and {self.G.number_of_edges()} edges"

    def __iter__(self):
        for sctid in self.G.nodes:
            yield self.get_concept_details(sctid)

    def get_children(self, sctid: int) -> List[SnomedConceptDetails]:
        return [
            r.src
            for r in self.__get_in_relationships(sctid)
            if r.type_id == SnomedGraph.is_a_relationship_typeId
        ]

    def get_parents(self, sctid: int) -> List[SnomedConceptDetails]:
        return [
            r.tgt
            for r in self.__get_out_relationships(sctid)
            if r.type_id == SnomedGraph.is_a_relationship_typeId
        ]

    def get_inferred_relationships(self, sctid: int) -> List[SnomedRelationshipGroup]:
        """
        Récupère les relations inférées d'un concept.
        (N.B. hors relations « is a », à récupérer via get_parents().)

        Args :
            sctid: Identifiant de concept SNOMED valide.
        Retourne :
            A list of SnomedRelationshipGroup objects.
        """
        inferred_relationships = [
            r
            for r in self.__get_out_relationships(sctid)
            if r.type_id != SnomedGraph.is_a_relationship_typeId
        ]
        key_ = lambda r: r.group
        inferred_relationships_grouped = groupby(sorted(inferred_relationships, key=key_), key=key_)
        inferred_relationship_groups = [
            SnomedRelationshipGroup(g, list(r)) for g, r in inferred_relationships_grouped
        ]
        return inferred_relationship_groups

    def get_concept_details(self, sctid: int) -> SnomedConceptDetails:
        """
        Récupère les détails de base d'un concept : SCTID, FSN et synonymes.

        Args :
            sctid: Identifiant de concept SNOMED valide.
        Retourne :
            Un objet SnomedConceptDetails.
        """
        return SnomedConceptDetails(sctid=sctid, **self.G.nodes[sctid])

    def get_full_concept(self, sctid: int) -> SnomedConcept:
        """
        Retrieve all attributes for a given concept.

        Args :
            sctid: Identifiant de concept SNOMED valide.
        Retourne :
            A SnomedConcept object.
        """
        concept_details = self.get_concept_details(sctid)
        parents = self.get_parents(sctid)
        children = self.get_children(sctid)
        inferred_relationship_groups = self.get_inferred_relationships(sctid)
        return SnomedConcept(concept_details, parents, children, inferred_relationship_groups)

    def __get_out_relationships(self, src_sctid: int) -> Generator[Dict, None, None]:
        src = SnomedConceptDetails(sctid=src_sctid, **self.G.nodes[src_sctid])
        for _, tgt_sctid in self.G.out_edges(src_sctid):
            tgt = SnomedConceptDetails(sctid=tgt_sctid, **self.G.nodes[tgt_sctid])
            vals = self.G.edges[(src_sctid, tgt_sctid)]
            yield SnomedRelationship(src, tgt, **vals)

    def __get_in_relationships(self, tgt_sctid: int) -> Generator[Dict, None, None]:
        tgt = SnomedConceptDetails(sctid=tgt_sctid, **self.G.nodes[tgt_sctid])
        for src_sctid, _ in self.G.in_edges(tgt_sctid):
            src = SnomedConceptDetails(sctid=src_sctid, **self.G.nodes[src_sctid])
            vals = self.G.edges[(src_sctid, tgt_sctid)]
            yield SnomedRelationship(src, tgt, **vals)

    def get_descendants(self, sctid: int, steps_removed: int = None) -> List[SnomedConceptDetails]:
        """
        Récupère les descendants d'un concept.

        Args :
            sctid: Identifiant de concept SNOMED valide.
            steps_removed: Nombre de niveaux à descendre dans la hiérarchie.
                           (1 => enfants ; 2 => enfants + petits-enfants, etc.)
                           si None alors tous children are retrieved.
        Retourne :
            A list containing the SCTIDs of all descendants.
        """
        if steps_removed is None:
            steps_removed = 99999
        elif steps_removed <= 0:
            raise AssertionError("steps_removed must be > 0 or None")
        children = self.get_children(sctid)
        descendants = set(children)
        if steps_removed > 1:
            for c in children:
                descendants = descendants.union(self.get_descendants(c.sctid, steps_removed - 1))
        return descendants

    def get_ancestors(self, sctid: int, steps_removed: int = None) -> List[SnomedConceptDetails]:
        """
        Récupère les ancêtres d'un concept.

        Args :
            sctid: Identifiant de concept SNOMED valide.
            steps_removed: Nombre de niveaux à remonter dans la hiérarchie.
                           (1 => parents ; 2 => parents + grands-parents, etc.)
                           si None alors tous parents are retrieved.
        Retourne :
            A list containing the SCTIDs of all descendants.
        """
        if steps_removed is None:
            steps_removed = 99999
        elif steps_removed <= 0:
            raise AssertionError("steps_removed must be > 0 or None")
        parents = self.get_parents(sctid)
        ancestors = set(parents)
        if steps_removed > 1:
            for p in parents:
                ancestors = ancestors.union(self.get_ancestors(p.sctid, steps_removed - 1))
        return set([a for a in ancestors if not a.sctid == SnomedGraph.root_concept_id])

    def get_neighbourhood(self, sctid: int, steps_removed: int = 1) -> List[SnomedConceptDetails]:
        """
        Récupère les voisins d'un concept.
        Les voisins incluent ancêtres, descendants et cousins jusqu'au degré donné.

        Args :
            sctid: Identifiant de concept SNOMED valide.
            steps_removed: Nombre de pas vers le haut ou le bas dans la hiérarchie.
                           Defaults to 1 (parents + children).
        Retourne :
            A list containing the SCTIDs of all neighbours.
        """
        assert steps_removed > 0
        parents = self.get_parents(sctid)
        children = self.get_children(sctid)
        neighbourhood = set(parents).union(children)
        if steps_removed > 1:
            for n in list(neighbourhood):
                neighbourhood = neighbourhood.union(
                    self.get_neighbourhood(n.sctid, steps_removed - 1)
                )
        neighbourhood = [
            n for n in neighbourhood if n.sctid not in [sctid, SnomedGraph.root_concept_id]
        ]
        return neighbourhood

    def find_path(self, sctid1: int, sctid2: int, print_: bool = False) -> List[SnomedRelationship]:
        """
        Détails d'un chemin éventuel entre deux concepts.
        Tous les types de relation sont considérés, mais seuls les vrais ancêtres ou
        descendants : les « cousins » ne donnent pas de chemin.

        Args :
            sctid1: Identifiant de concept SNOMED valide.
            sctid2: Identifiant de concept SNOMED valide.
            print_: Whether to print the full path as a string.
        Retourne :
            A list of Relationships of the form (source, relationship_type, target).
            These are the steps from source to target.
        """
        path = []
        if nx.has_path(self.G, sctid1, sctid2):
            nodes = nx.shortest_path(self.G, sctid1, sctid2)
        elif nx.has_path(self.G, sctid2, sctid1):
            nodes = nx.shortest_path(self.G, sctid2, sctid1)
        else:
            nodes = []
        for src_sctid, tgt_sctid in pairwise(nodes):
            vals = self.G.edges[(src_sctid, tgt_sctid)]
            src = SnomedConceptDetails(sctid=src_sctid, **self.G.nodes[src_sctid])
            tgt = SnomedConceptDetails(sctid=tgt_sctid, **self.G.nodes[tgt_sctid])
            relationship = SnomedRelationship(src, tgt, **vals)
            path.append(relationship)
        if print_:
            if len(nodes) > 0:
                str_ = f"[{path[0].src}]"
                for r in path:
                    str_ += f" ---[{r.type}]---> [{r.tgt}]"
                print(str_)
            else:
                print("No path found.")
        return path

    def save(self, path: str) -> None:
        """
        Enregistre ce SnomedGraph

        Args :
            path: chemin + nom du fichier de sortie.
        Retourne :
            None
        """
        nx.write_gml(self.G, path)

    def to_pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch node and edge CSVs for this graph

        Args :
            None
        Retourne :
            Two CSVs (nodes, edges) as Pandas DataFrames
        """
        nodes_df = pd.DataFrame([{"sctid": n, **self.G.nodes[n]} for n in self.G.nodes]).set_index(
            "sctid"
        )
        edges_df = nx.to_pandas_edgelist(self.G)
        return nodes_df, edges_df

    @property
    def relationship_types(self) -> Set[str]:
        """
        Ensemble des types de relation présents.

        Args :
            None
        Retourne :
            Un ensemble de chaînes
        """
        return set(nx.get_edge_attributes(self.G, "type").values())

    @staticmethod
    def from_serialized(path: str):
        """
        Charge un SnomedGraph depuis une sérialisation.

        Args :
            path: chemin + nom du fichier de sortie.
        Retourne :
            Un SnomedGraph
        """
        G = nx.read_gml(path, destringizer=int)
        return SnomedGraph(G)

    @staticmethod
    def from_rf2(path: str):
        """
        Crée un SnomedGraph depuis un chemin de release RF2 SNOMED.

        Args :
            path: Chemin du dossier de release RF2.
        Retourne :
            Un SnomedGraph
        """
        if path[-1] == "/":
            path = path[:-1]
        release_date_pattern = r"\d{8}"
        match = re.search(release_date_pattern, path)
        try:
            release_date = match.group(0)
        except AttributeError:
            raise AssertionError(
                f"The path does not appear to contain a valid SNOMED CT Release Format name."
            )
        else:
            # Charger les relations
            relationships_df = pd.read_csv(
                f"{path}/Snapshot/Terminology/sct2_Relationship_Snapshot_INT_{release_date}.txt",
                delimiter="\t",
            )
            relationships_df = relationships_df[relationships_df.active == 1]

            # Charger les concepts
            concepts_df = pd.read_csv(
                f"{path}/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_{release_date}.txt",
                delimiter="\t",
            )
            concepts_df = concepts_df[concepts_df.active == 1]
            concepts_df.set_index("conceptId", inplace=True)

            # Table de types de relations
            relationship_types = concepts_df.loc[relationships_df.typeId.unique()]
            relationship_types = relationship_types[
                relationship_types.typeId == SnomedGraph.fsn_typeId
            ]
            relationship_types = relationship_types.term.to_dict()

            # Initialiser le graphe
            n_concepts = concepts_df.shape[0]
            n_relationships = relationships_df.shape[0]
            print(
                f"{n_concepts} terms and {n_relationships} relationships were found in the release."
            )
            G = nx.DiGraph()

            # Créer les relations
            print("Creating Relationships...")
            for r in relationships_df.to_dict(orient="records"):
                G.add_edge(
                    r["sourceId"],
                    r["destinationId"],
                    group=r["relationshipGroup"],
                    type=relationship_types[r["typeId"]],
                    type_id=r["typeId"],
                )

            # Ajouter les concepts
            print("Adding Concepts...")
            for sctid, rows in concepts_df.groupby(concepts_df.index):
                synonyms = [
                    row.term for _, row in rows.iterrows() if row.typeId != SnomedGraph.fsn_typeId
                ]
                try:
                    fsn = rows[rows.typeId == SnomedGraph.fsn_typeId].term.values[0]
                except IndexError:
                    fsn = synonyms[0]
                    synonyms = synonyms[1:]
                    print(f"Concept with SCTID {sctid} has no FSN. Using synonym '{fsn}' instead.")
                G.add_node(sctid, fsn=fsn, synonyms=synonyms)

            # Retirer les isolés
            G.remove_nodes_from(list(nx.isolates(G)))

            # Initialiser la classe
            return SnomedGraph(G)
