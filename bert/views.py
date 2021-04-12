from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import BertConfig
# Create your views here.
import torch

def home_view(request, *args, **kwargs):
    print(args, kwargs)
    print(request.user)
    return render(request, "home.html", {})

class call_model(APIView):
    def get(self,request):
        if request.method == 'GET':
            # sentence is the query we want to get the prediction for
            sent = request.GET.get('sentence')
            sent2 = request.GET.get('sentence2')
            task = request.GET.get('task')
            print(sent)
            print(sent2)
            print(task)
            have_two = 1
            if task in ["CoLA"]:
                have_two = 0

            if task in ["CoLA"]:
                tokenize_sent = BertConfig.tokenizer.encode_plus(
                    sent,
                    return_tensors = "pt"
                )
                # predict method used to get the prediction
                with torch.no_grad():
                    output = BertConfig.model(
                        tokenize_sent['input_ids'],
                        tokenize_sent['attention_mask'],
                        tokenize_sent['token_type_ids']
                    )
                    label =  {
                            0:"unacceptable",
                            1:"acceptable"
                    }
                    
                    response = {
                        "sentence": sent,
                        "sentence2": sent2,
                        "task": task,
                        "have_two": have_two,
                        "result": label[int(output[0].squeeze().argmax().item())],
                        "result_str": "Our prediction is " + label[int(output[0].squeeze().argmax().item())]
                    }

                # returning JSON response
                #return JsonResponse(response)

            else:
                tokenize_sent = BertConfig.tokenizer_MNLI.encode_plus(
                    sent,sent2,
                    return_tensors = "pt"
                )

                with torch.no_grad():
                    output = BertConfig.model_MNLI(
                        tokenize_sent['input_ids'],
                        tokenize_sent['attention_mask'],
                        tokenize_sent['token_type_ids']
                    )
                    label =  {
                            0:"entailment",
                            1:"neutral",
                            2:"contradiction",
                    }
                    response = {
                        "sentence": sent,
                        "sentence2": sent2,
                        "task": task,
                        "have_two": have_two,
                        "result": label[int(output[0].squeeze().argmax().item())],
                        "result_str": "Our prediction is " + label[int(output[0].squeeze().argmax().item())]
                    }

            return render(request, "res.html", response)