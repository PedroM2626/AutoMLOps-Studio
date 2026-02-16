"""
Views da API Django para o projeto Senti-Pred.
"""
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
import json

# Caminho para o modelo treinado
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'sentiment_model.pkl')


class SentimentPredictionView(APIView):
    """
    API para predição de sentimentos.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Carregar o modelo se existir
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = None
    
    def post(self, request):
        """
        Endpoint para predição de sentimentos.
        
        Espera um JSON com o campo 'text' contendo o texto para análise.
        Retorna a predição de sentimento e as probabilidades.
        """
        if self.model is None:
            return Response(
                {"error": "Modelo não encontrado. Treine o modelo primeiro."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        try:
            # Obter texto da requisição
            data = json.loads(request.body)
            text = data.get('text', '')
            
            if not text:
                return Response(
                    {"error": "O campo 'text' é obrigatório."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Fazer predição
            sentiment = self.model.predict([text])[0]
            
            # Obter probabilidades se disponível
            try:
                probabilities = self.model.predict_proba([text])[0].tolist()
                classes = self.model.classes_.tolist()
                probs_dict = {str(cls): prob for cls, prob in zip(classes, probabilities)}
            except:
                probs_dict = {}
            
            # Retornar resultado
            return Response({
                "text": text,
                "sentiment": sentiment,
                "probabilities": probs_dict
            })
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ModelInfoView(APIView):
    """
    API para informações sobre o modelo.
    """
    
    def get(self, request):
        """
        Retorna informações sobre o modelo carregado.
        """
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            
            # Extrair informações do modelo
            model_type = type(model).__name__
            
            # Verificar se é um pipeline
            if hasattr(model, 'steps'):
                steps = [step[0] for step in model.steps]
                classifier = type(model.steps[-1][1]).__name__
            else:
                steps = []
                classifier = model_type
            
            return Response({
                "model_loaded": True,
                "model_type": model_type,
                "pipeline_steps": steps,
                "classifier": classifier,
                "model_path": MODEL_PATH
            })
        else:
            return Response({
                "model_loaded": False,
                "error": "Modelo não encontrado"
            })


def health_check(request):
    """
    Endpoint simples para verificar se a API está funcionando.
    """
    return JsonResponse({"status": "ok"})