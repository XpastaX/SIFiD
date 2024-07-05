from data.dataloader import SummaCBenchmark
from model.imager import SummaCImager
import torch
from tqdm import tqdm


def extract(cut, imager: SummaCImager, model_name:str):
    assert cut in ["val", "test"]
    data = SummaCBenchmark(cut=cut)
    datasets = {item['name']: item['dataset'] for item in data.datasets}
    for name in datasets:
        print(f"Processing {name}")
        for sample in tqdm(datasets[name]):
            doc = sample['document']
            sum = sample['claim']
            image_ent, image_cnt, doc_sent, sum_sent, cos_similarity = imager.build_image(doc, sum)
            assert image_ent.shape == (len(doc_sent), len(sum_sent))
            assert len(image_ent.shape) == 2
            entail = image_ent.max(-1)
            similar = cos_similarity.max(-1)
            sample['image_ent'] = image_ent
            sample['image_cnt'] = image_cnt
            sample['similarity'] = cos_similarity
            sample['entail'] = entail
            sample['sentiment'] = similar
            sample['doc_sent'] = doc_sent
            sample['sum_sent'] = sum_sent
    torch.save(datasets, f'data/processed/SummaC_{model_name}_{cut}.torch')


def extract_from_SummaC(model_name="vitc", device='cpu'):
    imager = SummaCImager(model_name_nli=model_name, device=device)
    for cut in ['test']:
        extract(cut, imager, model_name)


if __name__ == "__main__":
    extract_from_SummaC(model_name="vitc", device='cuda')