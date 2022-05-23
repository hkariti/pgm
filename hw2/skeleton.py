from pgmpy.models.MarkovNetwork import MarkovNetwork
from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor


def Q1():
    G = MarkovNetwork()
    G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    G.add_edges_from([('a', 'b'), ('a', 'c'), ('a', 'g'), ('b', 'c'), ('b', 'h'), ('c', 'd'), ('c', 'j'), ('d', 'e'),
                      ('e', 'f'), ('e', 'i'), ('e', 'j'), ('f', 'g'), ('f', 'j'), ('g', 'h'), ('g', 'i'), ('h', 'i')])

    ############################################################
    '''Define the node factors of the model'''
    colors_node_vals = {'pink': [3, 4], 'blue': [4, 1], 'yellow': [1, 3]}
    node_factors = [DiscreteFactor(['a'], [2], colors_node_vals['yellow']),
                    DiscreteFactor(['b'], [2], colors_node_vals['blue']),
                    DiscreteFactor(['c'], [2], colors_node_vals['pink']),
                    DiscreteFactor(['d'], [2], colors_node_vals['pink']),
                    DiscreteFactor(['e'], [2], colors_node_vals['yellow']),
                    DiscreteFactor(['f'], [2], colors_node_vals['blue']),
                    DiscreteFactor(['g'], [2], colors_node_vals['pink']),
                    DiscreteFactor(['h'], [2], colors_node_vals['yellow']),
                    DiscreteFactor(['i'], [2], colors_node_vals['pink']),
                    DiscreteFactor(['j'], [2], colors_node_vals['blue'])]

    G.add_factors(*node_factors)

    ############################################################

    ############################################################
    '''Define the edge factors of the model'''
    edge_fact_vals = {"pink-yellow": [2, 7, 7, 2], "pink-blue": [1, 8, 8, 1],
                      "yellow-blue": [1, 3, 3, 1], "same": [5, 1, 1, 5]}
    edge_factors = [DiscreteFactor(['a', 'b'], [2, 2], edge_fact_vals['yellow-blue']),
                    DiscreteFactor(['c', 'b'], [2, 2], edge_fact_vals['pink-blue']),
                    DiscreteFactor(['c', 'a'], [2, 2], edge_fact_vals['pink-yellow']),
                    DiscreteFactor(['h', 'b'], [2, 2], edge_fact_vals['yellow-blue']),
                    DiscreteFactor(['c', 'd'], [2, 2], edge_fact_vals['same']),
                    DiscreteFactor(['c', 'j'], [2, 2], edge_fact_vals['pink-blue']),
                    DiscreteFactor(['g', 'a'], [2, 2], edge_fact_vals['pink-yellow']),
                    DiscreteFactor(['d', 'e'], [2, 2], edge_fact_vals['pink-yellow']),
                    DiscreteFactor(['e', 'j'], [2, 2], edge_fact_vals['yellow-blue']),
                    DiscreteFactor(['e', 'f'], [2, 2], edge_fact_vals['yellow-blue']),
                    DiscreteFactor(['i', 'e'], [2, 2], edge_fact_vals['pink-yellow']),
                    DiscreteFactor(['g', 'f'], [2, 2], edge_fact_vals['pink-blue']),
                    DiscreteFactor(['f', 'j'], [2, 2], edge_fact_vals['same']),
                    DiscreteFactor(['g', 'i'], [2, 2], edge_fact_vals['same']),
                    DiscreteFactor(['g', 'h'], [2, 2], edge_fact_vals['pink-yellow']),
                    DiscreteFactor(['i', 'h'], [2, 2], edge_fact_vals['pink-yellow'])]

    G.add_factors(*edge_factors)
    ############################################################

    ############################################################
    '''1.1) Present one factor for every pair of colors'''
    ############################################################
    print(f"blue-blue:")
    # Complete blue-blue
    print(edge_factors[-4])

    print(f"pink-pink:")
    # Complete pink-pink
    print(edge_factors[-3])

    print(f"blue-pink:")
    # Complete pink-
    print(edge_factors[1])

    print(f"blue-yellow:")
    # Complete pink-pink
    print(edge_factors[0])

    print(f"pink-yellow:")
    # Complete pink-pink
    print(edge_factors[-1])

    ############################################################
    '''1.2) Implement inference over G'''
    ############################################################

    # Complete inference implementation
    belief_prop = BeliefPropagation(G)

    # find probability that at least one yellow buys the product
    print('1.2.1) At least one yellow:')
    # complete
    prob_y = belief_prop.query(variables=['a', 'h', 'e'], show_progress=False)
    # print(prob_y)
    print(1-prob_y.get_value(a=0, h=0, e=0))

    # find probability that at least two blues buy the product
    print('1.2.2) At least two blues:')
    # complete
    prob_b = belief_prop.query(variables=['b', 'j', 'f'], show_progress=False)
    print(1-(prob_b.get_value(b=0, j=0, f=0) + prob_b.get_value(b=1, j=0, f=0) + prob_b.get_value(b=0, j=1, f=0) +
             prob_b.get_value(b=0, j=0, f=1)))

    # most probable configuration
    print('1.2.3) Most likely configuration:')
    # complete
    config = belief_prop.map_query(list(G.nodes))
    print(config)

    # most probable configuration given that all the yellows bought the product
    print('1.2.4) Most likely configuration given that all the yellows bought the product:')
    # complete
    config_cond = belief_prop.map_query(list(set(G.nodes) - {'a', 'h', 'e'}), evidence={'a': 1, 'h': 1, 'e': 1},
                                        show_progress=False)
    print(config_cond)


def Q2():
    G = MarkovNetwork()
    G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    G.add_edges_from([('a', 'b'), ('a', 'c'), ('a', 'g'), ('b', 'c'), ('b', 'h'), ('c', 'd'), ('c', 'j'), ('d', 'e'),
                      ('e', 'f'), ('e', 'i'), ('e', 'j'), ('f', 'g'), ('f', 'j'), ('g', 'h'), ('g', 'i'), ('h', 'i')])

    ############################################################
    '''Define the node factors of the model'''
    ############################################################
    for node in G.nodes:
        node_fact = DiscreteFactor([node], [2], [0.5, 0.5])  # just put positive numbers
        G.add_factors(node_fact)
    ############################################################
    '''Define the edge factors of the model'''
    ############################################################
    for edge in G.edges:
        v0 = edge[0]
        v1 = edge[1]
        factor = DiscreteFactor([v0, v1], [2, 2], [1, 1, 1, 0])
        G.add_factors(factor)

    belief_prop = BeliefPropagation(G)
    num_ind_sets = belief_prop.query(variables=list(G.nodes), show_progress=False)
    print( "num of independent sets: ", (num_ind_sets.values > 0).sum())

    # Maximum Independent Set
    print('2.2) Find the size of the largest independent set in G:')
    # complete
    G = MarkovNetwork()
    G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    G.add_edges_from([('a', 'b'), ('a', 'c'), ('a', 'g'), ('b', 'c'), ('b', 'h'), ('c', 'd'), ('c', 'j'), ('d', 'e'),
                      ('e', 'f'), ('e', 'i'), ('e', 'j'), ('f', 'g'), ('f', 'j'), ('g', 'h'), ('g', 'i'), ('h', 'i')])

    for node in G.nodes:
        node_fact = DiscreteFactor([node], [2], [1, 2])  # just put positive numbers
        G.add_factors(node_fact)

    for edge in G.edges:
        v0 = edge[0]
        v1 = edge[1]
        factor = DiscreteFactor([v0, v1], [2, 2], [1, 1, 1, 0])
        G.add_factors(factor)

    belief_prop = BeliefPropagation(G)

    size_mis = sum(belief_prop.map_query(list(G.nodes), show_progress=False).values())
    config2 = belief_prop.map_query(list(G.nodes), show_progress=False)
    print(size_mis)
    print("independent set of max size: ", config2)


if __name__ == '__main__':
    Q1()
    Q2()
