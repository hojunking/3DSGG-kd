import torch.nn as nn
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from typing import Optional
from torch_scatter import scatter



class MessagePassing_VGfM(MessagePassing):
    def __init__(self, dim_node, aggr='mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.subj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.subj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.geo_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_feature, geo_feature, edge_index):
        node_msg, edge_msg = self.propagate(
            edge_index, x=x, edge_feature=edge_feature, geo_feature=geo_feature)
        return node_msg, edge_msg

    def message(self, x_i, x_j, edge_feature, geo_feature):
        message_pred_to_subj = self.subj_node_gate(
            torch.cat([x_i, edge_feature], dim=1)) * edge_feature  # n_rel x d
        message_pred_to_obj = self.obj_node_gate(
            torch.cat([x_j, edge_feature], dim=1)) * edge_feature  # n_rel x d
        node_message = (message_pred_to_subj+message_pred_to_obj)

        message_subj_to_pred = self.subj_edge_gate(
            torch.cat([x_i, edge_feature], 1)) * x_i  # nrel x d
        message_obj_to_pred = self.obj_edge_gate(
            torch.cat([x_j, edge_feature], 1)) * x_j  # nrel x d
        message_geo = self.geo_edge_gate(
            torch.cat([geo_feature, edge_feature], 1)) * geo_feature
        edge_message = (message_subj_to_pred+message_obj_to_pred+message_geo)

        # x = torch.cat([x_i,edge_feature,x_j],dim=1)
        # x = self.nn1(x)#.view(b,-1)
        # new_x_i = x[:,:self.dim_hidden]
        # new_e   = x[:,self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        # new_x_j = x[:,(self.dim_hidden+self.dim_edge):]
        # x = new_x_i+new_x_j
        return [node_message, edge_message]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)
        return x

class MessagePassing_Gate(MessagePassing):
    def __init__(self, dim_node, aggr='mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.temporal_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        x_i = self.temporal_gate(torch.cat([x_i, x_j], dim=1)) * x_i
        return x_i
    
    
class TripletVGfM(torch.nn.Module):
    def __init__(self, dim_node, num_layers, aggr='mean', **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dim_node = dim_node
        self.edge_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)
        self.node_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)

        self.msg_vgfm = MessagePassing_VGfM(dim_node=dim_node, aggr=aggr)
        self.msg_t_node = MessagePassing_Gate(dim_node=dim_node, aggr=aggr)
        self.msg_t_edge = MessagePassing_Gate(dim_node=dim_node, aggr=aggr)

        self.edge_encoder = ssg.models.edge_encoder.EdgeEncoder_VGfM()

        self.reset_parameter()

    def reset_parameter(self):
        pass
        # reset_parameters_with_activation(self.nn1[0], 'relu')
        # reset_parameters_with_activation(self.nn1[3], 'relu')
        # reset_parameters_with_activation(self.nn2[0], 'relu')

    def forward(self, data):
        '''shortcut'''
        x = data['roi'].x
        edge_feature = data['edge2D'].x
        edge_index = data['roi', 'to', 'roi'].edge_index
        geo_feature = data['roi'].desp
        temporal_node_graph = data['roi', 'temporal', 'roi'].edge_index
        temporal_edge_graph = data['edge2D', 'temporal', 'edge2D'].edge_index

        '''process'''
        x = self.node_gru(x)
        edge_feature = self.edge_gru(edge_feature)
        extended_geo_feature = self.edge_encoder(geo_feature, edge_index)
        for i in range(self.num_layers):
            node_msg, edge_msg = self.msg_vgfm(
                x=x, edge_feature=edge_feature, geo_feature=extended_geo_feature, edge_index=edge_index)
            if temporal_node_graph.shape[0] == 2:
                temporal_node_msg = self.msg_t_node(
                    x=x, edge_index=temporal_node_graph)
                node_msg += temporal_node_msg
            if temporal_edge_graph.shape[0] == 2:
                temporal_edge_msg = self.msg_t_edge(
                    x=edge_feature, edge_index=temporal_edge_graph)
                edge_msg += temporal_edge_msg
            x = self.node_gru(node_msg, x)
            edge_feature = self.edge_gru(edge_msg, edge_feature)

        return x, edge_feature