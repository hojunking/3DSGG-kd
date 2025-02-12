import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDimMapper(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super(FeatureDimMapper, self).__init__()
        # 1x1 Convolution을 사용하여 Teacher feature 차원을 Student feature 차원으로 변환
        self.conv1x1 = nn.Conv1d(in_channels=teacher_dim, out_channels=student_dim, kernel_size=1)

    def forward(self, teacher_feature):
        # teacher_feature의 차원이 2차원인 경우 -> (B, feature_dim) 형식일 때, 차원 추가
        teacher_feature = teacher_feature.unsqueeze(1)  # (B, feature_dim) -> (B, 1, feature_dim)
        teacher_feature = teacher_feature.permute(0, 2, 1)  # (B, N, teacher_dim) -> (B, teacher_dim, N)
        
        # Conv1d 적용: 입력이 (batch_size, teacher_dim, sequence_length)이어야 함
        teacher_mapped_feature = self.conv1x1(teacher_feature)
        
        # Conv1d 후 다시 (batch_size, sequence_length, student_dim)으로 변환
        teacher_mapped_feature = teacher_mapped_feature.permute(0, 2, 1)  # (B, student_dim, N) -> (B, N, student_dim)
        return teacher_mapped_feature.squeeze(1)
# 사용 예시
# teacher_gcn_obj_feature = torch.randn(32, 128, 512)  # (batch_size, num_points, teacher_dim)
# student_gcn_obj_feature = torch.randn(32, 128, 256)  # (batch_size, num_points, student_dim)

# # Teacher feature -> Student feature 차원으로 변환하는 1x1 Convolution Layer 적용
# conv1x1_dim_reducer = FeatureDimMapper(teacher_dim=512, student_dim=256)

# # Teacher feature를 Student feature 크기로 변환
# teacher_mapped_obj_feature = conv1x1_dim_reducer(teacher_gcn_obj_feature)

# # 출력 크기 확인
# print("Mapped Teacher feature size:", teacher_mapped_obj_feature.size())  # 예상: (32, 128, 256)
