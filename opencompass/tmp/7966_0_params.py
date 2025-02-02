datasets = [
    [
        dict(
            abbr='hellaswag',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict({
                        '0':
                        dict(round=[
                            dict(prompt='{query} {A}', role='HUMAN'),
                        ]),
                        '1':
                        dict(round=[
                            dict(prompt='{query} {B}', role='HUMAN'),
                        ]),
                        '2':
                        dict(round=[
                            dict(prompt='{query} {C}', role='HUMAN'),
                        ]),
                        '3':
                        dict(round=[
                            dict(prompt='{query} {D}', role='HUMAN'),
                        ])
                    }),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/hellaswag/hellaswag.jsonl',
            reader_cfg=dict(
                input_columns=[
                    'query',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='gold'),
            type='opencompass.datasets.hellaswagDataset_V3'),
        dict(
            abbr='ARC-c',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict(
                        A=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textA}', role='BOT'),
                        ]),
                        B=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textB}', role='BOT'),
                        ]),
                        C=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textC}', role='BOT'),
                        ]),
                        D=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textD}', role='BOT'),
                        ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/ARC/ARC-c/ARC-Challenge-Dev.jsonl',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'textA',
                    'textB',
                    'textC',
                    'textD',
                ],
                output_column='answerKey'),
            type='opencompass.datasets.ARCDataset'),
        dict(
            abbr='ARC-e',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict(
                        A=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textA}', role='BOT'),
                        ]),
                        B=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textB}', role='BOT'),
                        ]),
                        C=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textC}', role='BOT'),
                        ]),
                        D=dict(round=[
                            dict(
                                prompt='Question: {question}\nAnswer: ',
                                role='HUMAN'),
                            dict(prompt='{textD}', role='BOT'),
                        ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/ARC/ARC-e/ARC-Easy-Dev.jsonl',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'textA',
                    'textB',
                    'textC',
                    'textD',
                ],
                output_column='answerKey'),
            type='opencompass.datasets.ARCDataset'),
        dict(
            abbr='piqa',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict({
                        0:
                        'The following makes sense: \nQ: {goal}\nA: {sol1}\n',
                        1:
                        'The following makes sense: \nQ: {goal}\nA: {sol2}\n'
                    }),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/piqa',
            reader_cfg=dict(
                input_columns=[
                    'goal',
                    'sol1',
                    'sol2',
                ],
                output_column='label',
                test_split='validation'),
            type='opencompass.datasets.piqaDataset'),
        dict(
            abbr='winogrande',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict({
                        1:
                        dict(round=[
                            dict(prompt='Good sentence: {opt1}', role='HUMAN'),
                        ]),
                        2:
                        dict(round=[
                            dict(prompt='Good sentence: {opt2}', role='HUMAN'),
                        ])
                    }),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/winogrande',
            reader_cfg=dict(
                input_columns=[
                    'opt1',
                    'opt2',
                ], output_column='answer'),
            type='opencompass.datasets.winograndeDataset'),
        dict(
            abbr='commonsense_qa',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                ice_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=dict(
                            begin='</E>',
                            round=[
                                dict(
                                    prompt='Question: {question}\nAnswer: ',
                                    role='HUMAN'),
                                dict(prompt='{A}', role='BOT'),
                            ]),
                        B=dict(
                            begin='</E>',
                            round=[
                                dict(
                                    prompt='Question: {question}\nAnswer: ',
                                    role='HUMAN'),
                                dict(prompt='{B}', role='BOT'),
                            ]),
                        C=dict(
                            begin='</E>',
                            round=[
                                dict(
                                    prompt='Question: {question}\nAnswer: ',
                                    role='HUMAN'),
                                dict(prompt='{C}', role='BOT'),
                            ]),
                        D=dict(
                            begin='</E>',
                            round=[
                                dict(
                                    prompt='Question: {question}\nAnswer: ',
                                    role='HUMAN'),
                                dict(prompt='{D}', role='BOT'),
                            ]),
                        E=dict(
                            begin='</E>',
                            round=[
                                dict(
                                    prompt='Question: {question}\nAnswer: ',
                                    role='HUMAN'),
                                dict(prompt='{E}', role='BOT'),
                            ])),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            path='./data/commonsenseqa',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'A',
                    'B',
                    'C',
                    'D',
                    'E',
                ],
                output_column='answerKey',
                test_split='validation'),
            type='opencompass.datasets.commonsenseqaDataset'),
    ],
]
models = [
    dict(
        abbr='Llama-2-7b-hf_hf',
        batch_size=8,
        key_truncate_index='96',
        max_out_len=256,
        max_seq_len=None,
        mkv_path=
        '/liymai24/sjtu/bokai/LLaMA-Factory/saves/LLaMA-7B/distillation/redpajama/train_only_proj_32_64_64_eval',
        model_kwargs=dict(),
        pad_token_id=None,
        path='/liymai24/sjtu/bokai/PCA_kvcache/checkpoint/Llama-2-7b-hf',
        peft_kwargs=dict(),
        peft_path=None,
        run_cfg=dict(num_gpus=1),
        stop_words=[],
        tokenizer_kwargs=dict(),
        tokenizer_path=None,
        type='opencompass.models.huggingface_above_v4_33.HuggingFaceBaseModel',
        value_truncate_index='96'),
]
work_dir = 'outputs/default/20250202_093252'
