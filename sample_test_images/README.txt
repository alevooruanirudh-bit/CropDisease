Sample photos for testing the website's "Upload photo" button.

All 91 images come from the HELD-OUT TEST split (splits/web/test.csv): the
model never sees them during training, so they are a fair check.
The file name and answer_key.csv give the correct answer for each image
(the model only looks at the pixels, never at the file name).

1_one_per_plant                  one photo per plant (53 plants), mostly lab/plain-background
2_fruit_stem_tuber_other_parts   fruit, stem, tuber, rhizome, panicle, head, whole plant
3_real_field_photos              real field photos (PlantDoc etc.) - the hardest, most realistic test
4_not_a_plant                    non-plant images; the site should answer "not a plant"

Sources and licences for each image: see the source_dataset column in answer_key.csv
and catalog/datasets.yaml.
