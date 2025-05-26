import torch


class TextCollate:
    def __call__(self, batch):
        x_padded = []
        max_y_len = max([i[2].size(0) for i in batch])
        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        # Extract labels for returning
        labels = [item[1] for item in batch]

        for i in range(len(batch)):
            # Get the image tensor and ensure it has consistent dimensions
            img = batch[i][0]

            # Make sure tensor is in expected format before adding to list
            x_padded.append(img.unsqueeze(0))

            # Handle target sequence
            y = batch[i][2]
            y_padded[: y.size(0), i] = y

        # Concatenate along batch dimension - this will fail if tensors have inconsistent channels
        try:
            x_padded = torch.cat(x_padded)
        except RuntimeError as e:
            # Debug information
            shapes = [x.shape for x in x_padded]
            print(f"Error during tensor concatenation. Tensor shapes: {shapes}")
            raise e

        return x_padded, labels, y_padded
