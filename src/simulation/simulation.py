import random
import math
import pandas as pd
import os
from enum import Enum
import traceback
from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix

from src.distributed_matching.distributed_matching import distributed_max_weight_matching, \
    plot_bipartite_graph_with_matching, clean_output_folder


class DistributionType(Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'
    LOGNORMAL = 'log-normal'
    EXPONENTIAL = 'exponential'
    BETA = 'beta'
    MIXTURE = 'mixture'


def uplink_delay(I,  # Input data in bytes
                 d,  # Distance
                 B,  # Uplink channel bandwidth in Hz
                 P,  # Transmission power of user device in Watts
                 N_0=4e-21,  # Power of AWGN in Watts/Hz
                 alpha=3.4  # Path loss exponent (typical value for urban environments
                 ):
    return I / (B * np.log2(1 + (P / (N_0 * (d ** alpha)))))


def downlink_delay(O,  # Output data in bytes
                   d,  # Distance
                   B,  # Downlink channel bandwidth in Hz
                   P,  # Transmission power of server in Watts
                   N_0=4e-21,  # Power of AWGN in Watts/Hz
                   alpha=3.4  # Path loss exponent (typical value for urban environments
                   ):
    return O / (B * np.log2(1 + (P / (N_0 * (d ** alpha)))))


def computation_delay(D_mhz,
                      mu_ghz=5  # Computation capacity in GHz (5 billion cycles per second)
                      ):
    return (D_mhz * 1e6) / (mu_ghz * 1e9)


def communication_delay(st, latency_per_hop=5e-3):
    # seconds per hop
    return st * latency_per_hop


# Class representing a 2D point (for users and access points)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Class representing a User
class User:
    def __init__(self, x, y, user_id):
        self.task_request = None
        self.position = Point(x, y)
        self.user_id = user_id  # Unique ID for the user
        self.input_data_size = None  # I_i (in MB)
        self.output_data_size = None  # O_i (in MB)
        self.computation_demand = None  # D_i (in MHz)
        self.uplink_bandwidth = None  # B_i^u in Hz
        self.uplink_transmission_power = None  # P_i^{tra} in Watts
        self.delay_tolerance = None
        self.connected_aps = []  # To store access points the user can connect to
        self.best_ap = None  # To store the best access point for the user

    def add_access_points_in_range(self, access_points):
        """
        Get the best access point for the user based on the closest distance.
        """
        best_distance = float('inf')
        best_ap = None

        for ap in access_points:
            user_pos = np.array([self.position.x, self.position.y])
            ap_pos = np.array([ap.position.x, ap.position.y])
            distance = np.linalg.norm(user_pos - ap_pos)
            if distance <= ap.range:  # Check if the AP is within range
                self.connected_aps.append(ap)
                if distance < best_distance:
                    best_distance = distance
                    best_ap = ap

        self.best_ap = best_ap

    def get_access_points_in_range(self):
        return self.best_ap

    # Generate uplink bandwidth and transmission power
    def generate_uplink_params(self):
        """
        Generate uplink bandwidth and transmission power for the user.
        """
        # Uplink Bandwidth: 0.18 MHz - 20 MHz (log-normal distribution)
        B_u_min, B_u_max = 0.18, 20.0  # in Hz
        self.uplink_bandwidth = np.clip(np.random.lognormal(mean=np.log((B_u_min + B_u_max) / 2), sigma=0.8),
                                        B_u_min, B_u_max) * 1e6

        # Uplink Transmission Power: 0.1 W - 0.5 W (normal distribution)
        P_tra_min, P_tra_max = 0.1, 0.5  # in Watts
        self.uplink_transmission_power = np.clip(
            np.random.normal(loc=(P_tra_min + P_tra_max) / 2, scale=(P_tra_max - P_tra_min) / 6), P_tra_min,
            P_tra_max)

    # Generate task requests with specific distributions
    def generate_task_request(self, dist_type=DistributionType.UNIFORM):
        """
        Generate task requests for the user based on the distribution type.

        :param dist_type: str, type of distribution ('normal', 'log-normal', 'uniform', 'exponential', 'beta', 'poisson')
        """

        # Input Data Size: 500 KB - 5 MB
        I_min, I_max = 0.5, 5.0  # In MB
        # Output Data Size: 50 KB - 1 MB
        O_min, O_max = 0.05, 1.0  # In MB
        # Computation Demand: 100 MHz - 500 MHz
        D_min, D_max = 100, 500  # In MHz
        # Maximum Delay Tolerance: 1 ms - 5000 ms
        T_min, T_max = 1, 5000  # In ms

        if dist_type == DistributionType.MIXTURE:
            dist_types = list(DistributionType)
            dist_type = np.random.choice(dist_types)

        # Generate task request based on the specified distribution
        if dist_type == DistributionType.NORMAL:
            # Normal distribution for input, output, and computation demand
            input_size = np.clip(np.random.normal(loc=(I_min + I_max) / 2, scale=(I_max - I_min) / 6), I_min, I_max)
            output_size = np.clip(np.random.normal(loc=(O_min + O_max) / 2, scale=(O_max - O_min) / 6), O_min,
                                  O_max)
            comp_demand = np.clip(np.random.normal(loc=(D_min + D_max) / 2, scale=(D_max - D_min) / 6), D_min,
                                  D_max)
            delay_tolerance = np.clip(np.random.normal(loc=(T_min + T_max) / 2, scale=(T_max - T_min) / 6), T_min,
                                      T_max)

        elif dist_type == DistributionType.LOGNORMAL:
            # Log-normal distribution for tasks with skewed requirements
            input_size = np.clip(np.random.lognormal(mean=np.log((I_min + I_max) / 2), sigma=0.5), I_min, I_max)
            output_size = np.clip(np.random.lognormal(mean=np.log((O_min + O_max) / 2), sigma=0.5), O_min, O_max)
            comp_demand = np.clip(np.random.lognormal(mean=np.log((D_min + D_max) / 2), sigma=0.5), D_min, D_max)
            delay_tolerance = np.clip(np.random.lognormal(mean=np.log((T_min + T_max) / 2), sigma=0.5), T_min, T_max)

        elif dist_type == DistributionType.EXPONENTIAL:
            # Exponential distribution for tasks with a higher probability of smaller tasks
            input_size = np.clip(np.random.exponential(scale=(I_max - I_min) / 2), I_min, I_max)
            output_size = np.clip(np.random.exponential(scale=(O_max - O_min) / 2), O_min, O_max)
            comp_demand = np.clip(np.random.exponential(scale=(D_max - D_min) / 2), D_min, D_max)
            delay_tolerance = np.clip(np.random.exponential(scale=(T_max - T_min) / 2), T_min, T_max)

        elif dist_type == DistributionType.BETA:
            # Beta distribution, useful for tasks that exhibit bursts of larger or smaller demands
            input_size = np.clip(np.random.beta(a=2, b=5) * (I_max - I_min) + I_min, I_min, I_max)
            output_size = np.clip(np.random.beta(a=2, b=5) * (O_max - O_min) + O_min, O_min, O_max)
            comp_demand = np.clip(np.random.beta(a=2, b=5) * (D_max - D_min) + D_min, D_min, D_max)
            delay_tolerance = np.clip(np.random.beta(a=2, b=5) * (T_max - T_min) + T_min, T_min, T_max)

        else:  # 'uniform' by default
            # Uniform distribution for input, output, and computation demand
            input_size = np.random.uniform(I_min, I_max)
            output_size = np.random.uniform(O_min, O_max)
            comp_demand = np.random.uniform(D_min, D_max)
            delay_tolerance = np.random.uniform(T_min, T_max)

        self.input_data_size = int(input_size * 1e6)  # in bytes
        self.output_data_size = int(output_size * 1e6)  # in bytes
        self.computation_demand = comp_demand  # in MHz
        self.delay_tolerance = delay_tolerance * 1e-3

        # Save task request information
        self.task_request = {
            'input_size_MB': input_size,
            'output_size_MB': output_size,
            'comp_demand_MHz': comp_demand,
            'delay_tolerance': delay_tolerance
        }

    def __repr__(self):
        return f"{self.user_id}"

    def __hash__(self):
        return hash(f"{self.user_id}")

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.user_id == other.user_id
        )

    def __str__(self):
        return self.__repr__()

    @property
    def user_dict(self) -> Dict:
        return dict(x=self.position.x,
                    y=self.position.y,
                    user_id=self.user_id,
                    input_data_size=self.input_data_size,
                    output_data_size=self.output_data_size,
                    computation_demand=self.computation_demand,
                    uplink_bandwidth=self.uplink_bandwidth,
                    uplink_transmission_power=self.uplink_transmission_power,
                    delay_tolerance=self.delay_tolerance)


# Class representing an Access Point
class AccessPoint:
    def __init__(self, x, y, ap_id):
        self.position = Point(x, y)
        self.range = random.randint(10, 100)
        self.has_cloudlet = False  # Cloudlet property, initially False
        self.ap_id = ap_id  # Unique ID for the access point
        self.downlink_bandwidth = None
        self.downlink_transmission_power = None

    # Generate downlink bandwidth and transmission power
    def generate_downlink_params(self):
        """
        Generate downlink bandwidth and transmission power for the access point.
        """
        # Downlink Bandwidth: 1.4 MHz - 20 MHz (log-normal distribution)
        B_d_min, B_d_max = 1.4, 20.0  # in Hz
        self.downlink_bandwidth = np.clip(np.random.lognormal(mean=np.log((B_d_min + B_d_max) / 2), sigma=0.8),
                                          B_d_min, B_d_max) * 1e6

        # Downlink Transmission Power: 5 W - 40 W (normal distribution)
        P_tra_min, P_tra_max = 5, 40  # in Watts
        self.downlink_transmission_power = np.clip(
            np.random.normal(loc=(P_tra_min + P_tra_max) / 2, scale=(P_tra_max - P_tra_min) / 6), P_tra_min,
            P_tra_max)

    # Function to check if a user is within range
    def is_in_range(self, user):
        distance = math.sqrt((self.position.x - user.position.x) ** 2 +
                             (self.position.y - user.position.y) ** 2)
        return distance <= self.range

    def __repr__(self):
        return f"{self.ap_id}"

    def __hash__(self):
        return hash(f"{self.ap_id}")

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.ap_id == other.ap_id
        )

    def __str__(self):
        return self.__repr__()

    @property
    def ap_dict(self) -> Dict:
        return dict(x=self.position.x,
                    y=self.position.y,
                    user_id=self.ap_id,
                    range=self.range,
                    has_cloudlet=self.has_cloudlet)


# Class representing the simulation area
class Area:
    def __init__(self, width):
        self.width = width

    # Deploy n users randomly within the area
    def deploy_users(self, n):
        users = []
        for i in range(n):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.width)
            users.append(User(x, y, i))
        return users

    # Deploy m access points randomly within the area
    def deploy_access_points(self, m):
        access_points = []
        for i in range(m):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.width)
            access_points.append(AccessPoint(x, y, i))
        return access_points


# Main class for simulating the network
class NetworkSimulator:
    def __init__(self, w, n, m, p, u_sample):
        self.hop_distances = None
        self.width = w
        self.num_users = n
        self.num_ap = m
        self.prct_cl = p
        self.u_sample = u_sample
        self.area = Area(w)
        self.users = self.area.deploy_users(n)
        self.access_points = self.area.deploy_access_points(m)
        self.assign_cloudlets(p)
        self.connected_users = dict()

    # Function to randomly assign cloudlets to p% of access points
    def assign_cloudlets(self, p):
        num_cloudlets = int(len(self.access_points) * p / 100)
        cloudlet_aps = random.sample(self.access_points, num_cloudlets)
        for ap in cloudlet_aps:
            ap.has_cloudlet = True

    # Function to adjust the ranges of access points so every user is connected
    def adjust_ranges(self):
        added_users = set()

        # Step 1: Start with a very small range and gradually increase it
        range_increment = 1
        while len(added_users) < len(self.users):
            for ap in self.access_points:
                ap.range += range_increment + random.uniform(0,
                                                             math.ceil(self.width * 1.5 / 100.0))  # Increment the range
                for user in self.users:
                    if ap.is_in_range(user):
                        added_users.add(user)

    # Function to display which users are connected to which access points
    def display_connections(self):
        for i, ap in enumerate(self.access_points):
            self.connected_users[i] = []
            for j, user in enumerate(self.users):
                if ap.is_in_range(user):
                    self.connected_users[i].append(j)  # Collect user indices
            cloudlet_info = " (Cloudlet)" if ap.has_cloudlet else ""
            print(f"Access Point {i} (Range: {ap.range}){cloudlet_info} connects to Users: {self.connected_users[i]}")

    # Function to display task requests of users
    def display_task_requests(self):
        for user in self.users:
            print(f"User {user.user_id} -> Input Data: {user.input_data_size} Bytes, "
                  f"Output Data: {user.output_data_size} Bytes, "
                  f"Computation Demand: {user.computation_demand:.2f} MHz")

    # Function to plot the users, access points, and their ranges
    def plot_network(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot users as blue dots
        user_x = [user.position.x for user in self.users]
        user_y = [user.position.y for user in self.users]
        ax.scatter(user_x, user_y, c='blue', label='Users')

        # Variables to track if labels have been added
        cloudlet_label_added = False
        ap_label_added = False

        # Plot access points as red or green dots and their ranges as circles
        for ap in self.access_points:
            color = 'green' if ap.has_cloudlet else 'red'

            # Add label only once for each type of access point
            if ap.has_cloudlet and not cloudlet_label_added:
                ax.scatter(ap.position.x, ap.position.y, c=color, label='Access Point (Cloudlet)')
                cloudlet_label_added = True
            elif not ap.has_cloudlet and not ap_label_added:
                ax.scatter(ap.position.x, ap.position.y, c=color, label='Access Point')
                ap_label_added = True
            else:
                ax.scatter(ap.position.x, ap.position.y, c=color)

            # Draw range circle for the access point
            circle = plt.Circle((ap.position.x, ap.position.y), ap.range, color=color, fill=False, linestyle='--')
            ax.add_artist(circle)

        # Setting the plot limits
        ax.set_xlim(0, self.area.width)
        ax.set_ylim(0, self.area.width)
        ax.set_aspect('equal', 'box')

        # Labels and legend
        ax.set_title("User and Access Point Deployment")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

        # Show plot
        plt.show()

    # Function to create a graph using networkx
    def create_graph(self):
        G = nx.Graph()

        # Add users and access points as nodes
        for user in self.users:
            G.add_node(f"User_{user.user_id}", pos=(user.position.x, user.position.y), type='user')

        for ap in self.access_points:
            ap_type = 'cloudlet' if ap.has_cloudlet else 'ap'
            G.add_node(f"AP_{ap.ap_id}", pos=(ap.position.x, ap.position.y), type=ap_type)

        # Add edges if the user is in range of an access point
        for user in self.users:
            ap = user.get_access_points_in_range()
            if ap.is_in_range(user):
                G.add_edge(f"User_{user.user_id}", f"AP_{ap.ap_id}")

        return G

    # Function to plot the graph using networkx
    def plot_graph(self, G):
        pos = nx.get_node_attributes(G, 'pos')

        # Plot users in blue, access points in red, cloudlets in green
        node_colors = [
            'blue' if G.nodes[node]['type'] == 'user' else ('green' if G.nodes[node]['type'] == 'cloudlet' else 'red')
            for node in G.nodes]

        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=50, font_size=8, font_color="white")
        plt.title("User-Access Point Connectivity Graph")
        plt.show()

    # Create the backbone wired LAN graph connecting access points (sparse)
    def create_backbone_graph(self):
        backbone = nx.Graph()

        # Add access points as nodes
        for ap in self.access_points:
            ap_type = 'cloudlet' if ap.has_cloudlet else 'ap'
            backbone.add_node(f"AP_{ap.ap_id}", pos=(ap.position.x, ap.position.y), type=ap_type)

        # Create a distance matrix for the access points
        positions = [(ap.position.x, ap.position.y) for ap in self.access_points]
        dist_matrix = distance_matrix(positions, positions)

        # Create a minimum spanning tree (MST) for the backbone
        mst_edges = nx.minimum_spanning_tree(nx.Graph(dist_matrix)).edges

        # Add the MST edges to the backbone graph
        for i, j in mst_edges:
            ap1, ap2 = self.access_points[i], self.access_points[j]
            backbone.add_edge(f"AP_{ap1.ap_id}", f"AP_{ap2.ap_id}")

        # Add a few random edges to simulate redundancy
        extra_edges = random.sample(list(nx.non_edges(backbone)), k=min(5, len(list(nx.non_edges(backbone)))))
        for edge in extra_edges:
            backbone.add_edge(*edge)

        return backbone

    # Method to compute hop distances between access points in the backbone network
    def compute_shortest_hop_distances(self, backbone):
        hop_distances = {}

        # Iterate through each access point and calculate the shortest path to every other access point
        for ap1 in backbone.nodes:
            hop_distances[ap1] = {}
            lengths = nx.single_source_shortest_path_length(backbone, ap1)
            for ap2, hop_length in lengths.items():
                if 'AP' in ap2:  # Ensure we're calculating only between access points
                    hop_distances[ap1][ap2] = hop_length

        return hop_distances

    # Plot the wired backbone LAN network using networkx
    def plot_backbone_graph(self, backbone):
        pos = nx.get_node_attributes(backbone, 'pos')

        # Plot all access points (red for normal, green for cloudlets)
        node_colors = ['green' if backbone.nodes[node]['type'] == 'cloudlet' else 'red' for node in backbone.nodes]

        plt.figure(figsize=(8, 8))
        nx.draw(backbone, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8,
                font_color="white")
        plt.title("Backbone Wired LAN Network (Access Points)")
        plt.show()

    # Plot the combined wireless and backbone networks
    def plot_combined_network(self, wireless_graph, backbone_graph):
        pos = nx.get_node_attributes(wireless_graph, 'pos')

        plt.figure(figsize=(8, 8))

        # Plot the wireless graph
        nx.draw_networkx_edges(wireless_graph, pos, edgelist=wireless_graph.edges(), edge_color='gray', style='--',
                               alpha=0.4)
        nx.draw_networkx_nodes(wireless_graph, pos, node_color=[
            ('green' if wireless_graph.nodes[node]['type'] == 'cloudlet' else 'red') if 'AP' in node else 'blue' for
            node in wireless_graph.nodes], node_size=10)
        # nx.draw_networkx_labels(wireless_graph, pos, font_size=10, font_color='white')

        # Overlay the backbone network
        nx.draw_networkx_edges(backbone_graph, pos, edgelist=backbone_graph.edges(), edge_color='black', style='-',
                               alpha=0.8)
        nx.draw_networkx_nodes(backbone_graph, pos,
                               node_color=[('green' if backbone_graph.nodes[node]['type'] == 'cloudlet' else 'red')
                                           for node in backbone_graph.nodes], node_size=50, edgecolors='black')

        plt.title("Combined Wireless and Backbone Network")
        plt.savefig(f'./output_graphs/combined_network_{self.width}_{self.num_users}_{self.num_ap}_{self.prct_cl}_{self.u_sample}.pdf',
                    dpi=300)
        plt.close()

    def bipartite_reduction(self):
        B = nx.Graph()  # Initialize the bipartite graph

        # Set 'X' for users, and 'Y' for access points with cloudlets
        X = []
        Y = []
        W = {}
        AP_MAP = {}
        USER_MAP = {}
        REVERSE_AP_MAP = {}
        REVERSE_USER_MAP = {}
        ALL_MAP = {}
        REVERSE_ALL_MAP = {}
        DELAY_MAP = {}

        # Add users to set X
        count = 0
        for user in self.users:
            USER_MAP[user] = count
            ALL_MAP[user] = count
            REVERSE_ALL_MAP[count] = user
            REVERSE_USER_MAP[count] = user
            X.append(count)
            B.add_node(count, bipartite=0, property=user.user_dict)
            count += 1

        # Add access points with cloudlets to set Y
        for ap in self.access_points:
            if ap.has_cloudlet:  # Only include APs with cloudlets
                AP_MAP[ap] = count
                ALL_MAP[ap] = count
                REVERSE_ALL_MAP[count] = ap
                REVERSE_AP_MAP[count] = ap
                Y.append(count)
                B.add_node(count, bipartite=1, property=ap.ap_dict)
                count += 1

                # Add edges between sample of users and cloudlet-enabled APs
                num_sample_users = int(len(self.users) * self.u_sample / 100)
                sample_users = random.sample(self.users, num_sample_users)
                W[ap] = {}
                DELAY_MAP[ap] = {}
                for user in sample_users:
                    connected_ap: AccessPoint = user.get_access_points_in_range()
                    user_pos = np.array([user.position.x, user.position.y])
                    connected_ap_pos = np.array([connected_ap.position.x, connected_ap.position.y])
                    distance = np.linalg.norm(user_pos - connected_ap_pos)
                    upload_delay = uplink_delay(user.input_data_size,
                                                distance,
                                                user.uplink_bandwidth,
                                                user.uplink_transmission_power)
                    download_delay = downlink_delay(user.output_data_size,
                                                    distance,
                                                    connected_ap.downlink_bandwidth,
                                                    connected_ap.downlink_transmission_power)
                    compu_delay = computation_delay(user.computation_demand)
                    hop = self.hop_distances[f'AP_{connected_ap.ap_id}'][f'AP_{ap.ap_id}']
                    comm_delay = 2 * communication_delay(hop)
                    weight = 1.0 / (upload_delay + download_delay + compu_delay + comm_delay)
                    B.add_edge(USER_MAP[user], AP_MAP[ap], weight=weight)
                    W[ap][user] = weight
                    DELAY_MAP[ap][user] = (upload_delay + download_delay + compu_delay + comm_delay)

        # Add cloud node to set Y
        cloud = AccessPoint(-1, -1, len(self.access_points))
        AP_MAP[cloud] = count
        ALL_MAP[cloud] = count
        REVERSE_ALL_MAP[count] = cloud
        REVERSE_AP_MAP[count] = cloud
        Y.append(count)
        B.add_node(count, bipartite=1, property=cloud.ap_dict)
        count += 1
        W[cloud] = {}
        DELAY_MAP[cloud] = {}
        for user in self.users:
            connected_ap: AccessPoint = user.get_access_points_in_range()
            user_pos = np.array([user.position.x, user.position.y])
            connected_ap_pos = np.array([connected_ap.position.x, connected_ap.position.y])
            distance = np.linalg.norm(user_pos - connected_ap_pos)
            upload_delay = uplink_delay(user.input_data_size,
                                        distance,
                                        user.uplink_bandwidth,
                                        user.uplink_transmission_power)
            download_delay = downlink_delay(user.output_data_size,
                                            distance,
                                            connected_ap.downlink_bandwidth,
                                            connected_ap.downlink_transmission_power)
            compu_delay = computation_delay(user.computation_demand, mu_ghz=50)
            comm_delay = 2.0 * np.clip(np.random.normal(loc=0.3, scale=0.1), 0.1, 0.5)
            weight = 1.0 / (upload_delay + download_delay + compu_delay + comm_delay)
            B.add_edge(USER_MAP[user], AP_MAP[cloud], weight=weight)
            W[cloud][user] = weight
            DELAY_MAP[cloud][user] = (upload_delay + download_delay + compu_delay + comm_delay)

        return B, X, Y, W, cloud, USER_MAP, AP_MAP, REVERSE_USER_MAP, REVERSE_AP_MAP, ALL_MAP, REVERSE_ALL_MAP, DELAY_MAP

    # Plot the bipartite graph
    def plot_bipartite_graph(self, B, X, Y):
        pos = nx.bipartite_layout(B, X)

        plt.figure(figsize=(8, 8))

        # Draw users (from set X)
        nx.draw_networkx_nodes(B, pos, nodelist=X, node_color='blue', node_size=50, label='Users')

        # Draw access points with cloudlets (from set Y)
        nx.draw_networkx_nodes(B, pos, nodelist=Y, node_color='green', node_size=50, label='Cloudlet-Enabled APs')

        # Draw edges with edge weights
        edges = B.edges(data=True)
        edge_weights = [d['weight'] for (u, v, d) in edges]
        nx.draw_networkx_edges(B, pos, edgelist=edges, width=[math.log(w) for w in edge_weights], edge_color='gray')

        # Add labels
        nx.draw_networkx_labels(B, pos, font_size=8, font_color='white')

        plt.title("Bipartite Graph: Users and Cloudlet-Enabled Access Points")
        plt.legend(loc='upper left')
        plt.show()

    # Plot distribution of task requests over all users
    def plot_task_request_distribution(self):
        """
        Plot the distribution of input size, output size, and computation demand for all users.
        """
        # Gather data
        input_sizes = [user.task_request['input_size_MB'] for user in self.users]
        output_sizes = [user.task_request['output_size_MB'] for user in self.users]
        computation_demands = [user.task_request['comp_demand_MHz'] for user in self.users]

        # Create subplots for input size, output size, and computation demand
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot input size distribution
        sns.histplot(input_sizes, kde=True, ax=axs[0], color='blue')
        axs[0].set_title('Input Data Size Distribution')
        axs[0].set_xlabel('Input Size (MB)')
        axs[0].set_ylabel('Density')

        # Plot output size distribution
        sns.histplot(output_sizes, kde=True, ax=axs[1], color='green')
        axs[1].set_title('Output Data Size Distribution')
        axs[1].set_xlabel('Output Size (MB)')
        axs[1].set_ylabel('Density')

        # Plot computation demand distribution
        sns.histplot(computation_demands, kde=True, ax=axs[2], color='red')
        axs[2].set_title('Computation Demand Distribution')
        axs[2].set_xlabel('Computation Demand (MHz)')
        axs[2].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def plot_uplink_distribution(self):
        """
        Plot the distribution of uplink bandwidth and transmission power for all users.
        """
        # Gather data
        uplink_bandwidths = [user.uplink_bandwidth for user in self.users]
        uplink_powers = [user.uplink_transmission_power for user in self.users]

        # Create subplots for uplink bandwidth and transmission power
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot uplink bandwidth distribution
        sns.histplot(uplink_bandwidths, kde=True, ax=axs[0], color='blue')
        axs[0].set_title('Uplink Bandwidth Distribution')
        axs[0].set_xlabel('Uplink Bandwidth (Hz)')
        axs[0].set_ylabel('Density')

        # Plot uplink transmission power distribution
        sns.histplot(uplink_powers, kde=True, ax=axs[1], color='green')
        axs[1].set_title('Uplink Transmission Power Distribution')
        axs[1].set_xlabel('Uplink Transmission Power (W)')
        axs[1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def plot_downlink_distribution(self):
        """
        Plot the distribution of downlink bandwidth and transmission power for all access points.
        """
        # Gather data
        downlink_bandwidths = [ap.downlink_bandwidth for ap in self.access_points]
        downlink_powers = [ap.downlink_transmission_power for ap in self.access_points]

        # Create subplots for downlink bandwidth and transmission power
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot downlink bandwidth distribution
        sns.histplot(downlink_bandwidths, kde=True, ax=axs[0], color='blue')
        axs[0].set_title('Downlink Bandwidth Distribution')
        axs[0].set_xlabel('Downlink Bandwidth (Hz)')
        axs[0].set_ylabel('Density')

        # Plot downlink transmission power distribution
        sns.histplot(downlink_powers, kde=True, ax=axs[1], color='green')
        axs[1].set_title('Downlink Transmission Power Distribution')
        axs[1].set_xlabel('Downlink Transmission Power (W)')
        axs[1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def greedy_matching(self, G):
        # Create a list of edges sorted by weight in descending order
        sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

        matched_nodes = set()  # To keep track of already matched nodes
        matching = []  # List to store the resulting matching

        for u, v, data in sorted_edges:
            # If neither u nor v are matched, add the edge to the matching
            if u not in matched_nodes and v not in matched_nodes:
                matching.append((u, v))
                matched_nodes.add(u)
                matched_nodes.add(v)

        return matching

    def compute_utility(self, M, W, REVERSE_ALL_MAP, DELAY_MAP, cloud):
        # First compute the utility of all assigned users
        assigned_users = []
        matching_utility = 0.0
        utility = 0.0
        above_delay_tolerance = 0
        above_delay_tolerance_matching = 0
        max_miss = -1
        miss_users = []
        for u, v in M:
            node_u = REVERSE_ALL_MAP[u]
            node_v = REVERSE_ALL_MAP[v]
            if isinstance(node_u, User):
                user = node_u
                ap = node_v
            else:
                user = node_v
                ap = node_u
            assigned_users.append(user)
            utility = utility + W[ap][user]
            matching_utility = utility
            if DELAY_MAP[ap][user] >= user.delay_tolerance:
                above_delay_tolerance += 1
                above_delay_tolerance_matching +=1
                diff = DELAY_MAP[ap][user] - user.delay_tolerance
                if max_miss < diff:
                    max_miss = diff
                    miss_users.append(user)

        # Then for each unassigned user, compute the cost of
        # Assigning them to the Cloud node
        for user in self.users:
            if user not in assigned_users:
                utility = utility + W[cloud][user]
                if DELAY_MAP[cloud][user] >= user.delay_tolerance:
                    above_delay_tolerance += 1
                    diff = DELAY_MAP[cloud][user] - user.delay_tolerance
                    if max_miss < diff:
                        max_miss = diff
                        miss_users.append(user)

        pct_above = 100.0 * above_delay_tolerance / self.num_users
        pct_above_matching = 100.0 * above_delay_tolerance_matching / len(assigned_users)
        miss_user_input_data_size = []
        miss_user_output_data_size = []
        miss_user_computation_demand = []
        miss_user_delay_tolerance = []
        for user in miss_users:
            miss_user_input_data_size.append(user.input_data_size)
            miss_user_output_data_size.append(user.output_data_size)
            miss_user_computation_demand.append(user.computation_demand)
            miss_user_delay_tolerance.append(user.delay_tolerance)

        mean_miss_user_input_data_size = np.mean(miss_user_input_data_size)
        mean_miss_user_output_data_size = np.mean(miss_user_output_data_size)
        mean_miss_user_computation_demand = np.mean(miss_user_computation_demand)
        mean_miss_user_delay_tolerance = np.mean(miss_user_delay_tolerance)

        return (utility,
                matching_utility,
                pct_above_matching,
                pct_above,
                max_miss,
                mean_miss_user_input_data_size,
                mean_miss_user_output_data_size,
                mean_miss_user_computation_demand,
                mean_miss_user_delay_tolerance
                )

    # Run the simulation
    def run_simulation(self, dist_type=DistributionType.UNIFORM):
        # Generate task requests for users
        for user in self.users:
            user.generate_task_request(dist_type=dist_type)
            user.generate_uplink_params()

        #self.plot_task_request_distribution()
        #self.plot_uplink_distribution()

        for ap in self.access_points:
            ap.generate_downlink_params()

        #self.plot_downlink_distribution()

        self.adjust_ranges()

        # assign the best AP to user
        for user in self.users:
            user.add_access_points_in_range(self.access_points)

        #self.plot_network()  # Plot user and access point deployment

        # Create and plot the network graph
        G = self.create_graph()
        #self.plot_graph(G)

        A = self.create_backbone_graph()
        #self.plot_backbone_graph(A)

        # Compute hop distances between access points
        self.hop_distances = self.compute_shortest_hop_distances(A)

        self.plot_combined_network(G, A)

        [B, X, Y, W,
         cloud, USER_MAP,
         AP_MAP, REVERSE_USER_MAP,
         REVERSE_AP_MAP, ALL_MAP,
         REVERSE_ALL_MAP, DELAY_MAP] = self.bipartite_reduction()

        #self.plot_bipartite_graph(B, X, Y)
        M = distributed_max_weight_matching(B)
        #plot_bipartite_graph_with_matching(B,M)
        M_greedy = self.greedy_matching(B)

        (utility,
         matching_utility,
         pct_above_matching,
         pct_above,
         max_miss,
         mean_miss_user_input_data_size,
         mean_miss_user_output_data_size,
         mean_miss_user_computation_demand,
         mean_miss_user_delay_tolerance) = self.compute_utility(M, W, REVERSE_ALL_MAP, DELAY_MAP, cloud)

        (utility_greedy,
         matching_utility_greedy,
         pct_above_matching_greedy,
         pct_above_greedy,
         max_miss_greedy,
         mean_miss_user_input_data_size_greedy,
         mean_miss_user_output_data_size_greedy,
         mean_miss_user_computation_demand_greedy,
         mean_miss_user_delay_tolerance_greedy) = self.compute_utility(M_greedy, W, REVERSE_ALL_MAP, DELAY_MAP, cloud)


        print(f'Utility {utility}, pct_above: {pct_above}, utility_greedy: {utility_greedy}, pct_above_greedy: {pct_above_greedy}')
        return (utility,
                matching_utility,
                pct_above_matching,
                pct_above,
                max_miss,
                mean_miss_user_input_data_size,
                mean_miss_user_output_data_size,
                mean_miss_user_computation_demand,
                mean_miss_user_delay_tolerance,
                utility_greedy,
                matching_utility_greedy,
                pct_above_matching_greedy,
                pct_above_greedy,
                max_miss_greedy,
                mean_miss_user_input_data_size_greedy,
                mean_miss_user_output_data_size_greedy,
                mean_miss_user_computation_demand_greedy,
                mean_miss_user_delay_tolerance_greedy
                )



class NetworkSimulatorRunner:
    def __init__(self, csv_file='simulation_results.csv'):
        self.csv_file = csv_file
        self.parameters = []
        self.results = []
        self.failed_run = None

        # Ensure CSV file exists with correct headers
        if not os.path.exists(self.csv_file):
            self.init_csv()

    def init_csv(self):
        """ Initialize the CSV file with headers """
        df = pd.DataFrame(columns=['w',
                                   'n',
                                   'm',
                                   'p',
                                   'u_sample',
                                   'dist_type',
                                   'utility',
                                   'matching_utility',
                                   'pct_above_matching',
                                   'pct_above',
                                   'max_miss',
                                   'mean_miss_user_input_data_size',
                                   'mean_miss_user_output_data_size',
                                   'mean_miss_user_computation_demand',
                                   'mean_miss_user_delay_tolerance',
                                   'utility_greedy',
                                   'matching_utility_greedy',
                                   'pct_above_matching_greedy',
                                   'pct_above_greedy',
                                   'max_miss_greedy',
                                   'mean_miss_user_input_data_size_greedy',
                                   'mean_miss_user_output_data_size_greedy',
                                   'mean_miss_user_computation_demand_greedy',
                                   'mean_miss_user_delay_tolerance_greedy',
                                   ])
        df.to_csv(self.csv_file, index=False)

    def load_previous_runs(self):
        """ Load previous results from CSV file to resume from the last failure point """
        if os.path.exists(self.csv_file):
            self.results = pd.read_csv(self.csv_file).to_dict('records')

    def save_results(self,
                     parameters,
                     utility,
                     matching_utility,
                     pct_above_matching,
                     pct_above,
                     max_miss,
                     mean_miss_user_input_data_size,
                     mean_miss_user_output_data_size,
                     mean_miss_user_computation_demand,
                     mean_miss_user_delay_tolerance,
                     utility_greedy,
                     matching_utility_greedy,
                     pct_above_matching_greedy,
                     pct_above_greedy,
                     max_miss_greedy,
                     mean_miss_user_input_data_size_greedy,
                     mean_miss_user_output_data_size_greedy,
                     mean_miss_user_computation_demand_greedy,
                     mean_miss_user_delay_tolerance_greedy):
        """ Save simulation results to CSV """
        self.results.append({**parameters,
                             'utility': utility,
                             'matching_utility': matching_utility,
                             'pct_above_matching': pct_above_matching,
                             'pct_above': pct_above,
                             'max_miss': max_miss,
                             'mean_miss_user_input_data_size': mean_miss_user_input_data_size,
                             'mean_miss_user_output_data_size': mean_miss_user_output_data_size,
                             'mean_miss_user_computation_demand': mean_miss_user_computation_demand,
                             'mean_miss_user_delay_tolerance': mean_miss_user_delay_tolerance,
                             'utility_greedy': utility_greedy,
                             'matching_utility_greedy': matching_utility_greedy,
                             'pct_above_matching_greedy': pct_above_matching_greedy,
                             'pct_above_greedy': pct_above_greedy,
                             'max_miss_greedy': max_miss_greedy,
                             'mean_miss_user_input_data_size_greedy': mean_miss_user_input_data_size_greedy,
                             'mean_miss_user_output_data_size_greedy': mean_miss_user_output_data_size_greedy,
                             'mean_miss_user_computation_demand_greedy': mean_miss_user_computation_demand_greedy,
                             'mean_miss_user_delay_tolerance_greedy': mean_miss_user_delay_tolerance_greedy
                             })
        df = pd.DataFrame(self.results)
        df.to_csv(self.csv_file, index=False)

    def calculate_access_point_range(self, w, n):
        """
        Calculate the range of access points (m) based on the area size (w) and number of users (n).
        """
        area_sq_km = (w ** 2) / 1e6  # Convert w^2 to square kilometers
        user_density = n / area_sq_km  # Users per square kilometer

        # Access point density based on IoT research (low, moderate, high)
        if user_density < 1000:
            ap_range = (1 * area_sq_km, 5 * area_sq_km)  # Low density
        elif 1000 <= user_density < 10000:
            ap_range = (5 * area_sq_km, 30 * area_sq_km)  # Moderate density
        else:
            ap_range = (30 * area_sq_km, 60 * area_sq_km)  # High density

        return int(ap_range[0]), int(ap_range[1])

    def generate_combinations(self):
        """ Generate all combinations of parameters """
        w_range = range(2500, 10001, 2500)
        n_range = [w ** 2 * density // 1000000 for w in w_range for density in range(250, 4001, 500)]
        p_range = [60]
        u_sample_range = [50, 100]
        dist_types = list(DistributionType)

        # Generate combinations based on user density and w
        for w in w_range:
            for n in n_range:
                # Calculate the range of access points (m) based on IoT research
                min_m, max_m = self.calculate_access_point_range(w, n)
                for m in range(min_m, max_m + 1, (max_m - min_m) // 10):  # Step by 10% of the range
                    for p in p_range:
                        for u_sample in u_sample_range:
                            for dist_type in dist_types:
                                self.parameters.append((w, n, m, p, u_sample, dist_type))

    def run_simulation(self, w, n, m, p, u_sample, dist_type):
        """
        Simulate the network and compute utility.
        The logic for the actual simulation goes here.
        """
        print(f"Running simulation with w={w}, n={n}, m={m}, p={p}, u_sample={u_sample}, dist_type={dist_type}")
        # Create a NetworkSimulator instance with these parameters
        simulator = NetworkSimulator(w, n, m, p, u_sample)
        # Run the simulation
        (utility,
                matching_utility,
                pct_above_matching,
                pct_above,
                max_miss,
                mean_miss_user_input_data_size,
                mean_miss_user_output_data_size,
                mean_miss_user_computation_demand,
                mean_miss_user_delay_tolerance,
                utility_greedy,
                matching_utility_greedy,
                pct_above_matching_greedy,
                pct_above_greedy,
                max_miss_greedy,
                mean_miss_user_input_data_size_greedy,
                mean_miss_user_output_data_size_greedy,
                mean_miss_user_computation_demand_greedy,
                mean_miss_user_delay_tolerance_greedy) = simulator.run_simulation(dist_type=dist_type)

        return (utility,
                matching_utility,
                pct_above_matching,
                pct_above,
                max_miss,
                mean_miss_user_input_data_size,
                mean_miss_user_output_data_size,
                mean_miss_user_computation_demand,
                mean_miss_user_delay_tolerance,
                utility_greedy,
                matching_utility_greedy,
                pct_above_matching_greedy,
                pct_above_greedy,
                max_miss_greedy,
                mean_miss_user_input_data_size_greedy,
                mean_miss_user_output_data_size_greedy,
                mean_miss_user_computation_demand_greedy,
                mean_miss_user_delay_tolerance_greedy)

    def start(self):
        clean_output_folder()
        """ Start the simulations with all combinations and save results """
        self.load_previous_runs()  # Load any previous results

        for idx, params in enumerate(self.parameters):
            # Skip if the simulation has already been run (check by w, n, m, p, u_sample, dist_type)
            if idx < len(self.results):
                continue

            try:
                w, n, m, p, u_sample, dist_type = params
                (utility,
                matching_utility,
                pct_above_matching,
                pct_above,
                max_miss,
                mean_miss_user_input_data_size,
                mean_miss_user_output_data_size,
                mean_miss_user_computation_demand,
                mean_miss_user_delay_tolerance,
                utility_greedy,
                matching_utility_greedy,
                pct_above_matching_greedy,
                pct_above_greedy,
                max_miss_greedy,
                mean_miss_user_input_data_size_greedy,
                mean_miss_user_output_data_size_greedy,
                mean_miss_user_computation_demand_greedy,
                mean_miss_user_delay_tolerance_greedy) = self.run_simulation(w, n, m, p, u_sample, dist_type)
                self.save_results({
                    'w': w,
                    'n': n,
                    'm': m,
                    'p': p,
                    'u_sample': u_sample,
                    'dist_type': dist_type.value
                }, utility,
                matching_utility,
                pct_above_matching,
                pct_above,
                max_miss,
                mean_miss_user_input_data_size,
                mean_miss_user_output_data_size,
                mean_miss_user_computation_demand,
                mean_miss_user_delay_tolerance,
                utility_greedy,
                matching_utility_greedy,
                pct_above_matching_greedy,
                pct_above_greedy,
                max_miss_greedy,
                mean_miss_user_input_data_size_greedy,
                mean_miss_user_output_data_size_greedy,
                mean_miss_user_computation_demand_greedy,
                mean_miss_user_delay_tolerance_greedy)

            except Exception as e:
                print(f"Simulation failed for parameters: {params}")
                print(f"Error: {traceback.format_exc()}")
                self.failed_run = params
                break

        if self.failed_run:
            print(f"Simulation stopped at: {self.failed_run}. You can modify and restart from this point.")


sim_runner = NetworkSimulatorRunner()
sim_runner.generate_combinations()  # Generates all parameter combinations
sim_runner.start()  # Starts the simulation and stores results in the CSV
