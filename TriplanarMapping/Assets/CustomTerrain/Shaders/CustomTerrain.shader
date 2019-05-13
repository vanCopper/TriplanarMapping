Shader "Copper/Terrain/TriplanarMapping" 
{
    Properties 
    {  
        _Basemap("Base map (RGB)", 2D ) = "white" {}
    }
	
    SubShader {  
        Tags{"RenderType" = "Opaque" "RenderPipeline" = "LightweightPipeline"} 
           
        Pass {  
            Tags { "LightMode" = "LightweightForward" }  
               
            HLSLPROGRAM  
            #pragma target 4.6

			#pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.lightweight/ShaderLibrary/Core.hlsl"
			#include "Packages/com.unity.render-pipelines.lightweight/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.lightweight/ShaderLibrary/Shadows.hlsl"
			
            sampler2D _Basemap;
			half4 _Basemap_ST;
            
             
            struct a2v 
            {  
                half4 vertex : POSITION;  
                half3 normal : NORMAL;  
				half2 texcoord : TEXCOORD0;
            };

            struct v2f 
			{   
                half4 position      : SV_POSITION;  
				half3 worldNormal		: NORMAL;
				half2 texcoord : TEXCOORD1;
                half3 worldPos : TEXCOORD2;
            }; 
  
            v2f vert(a2v v) 
            {  
                v2f o;
				VertexPositionInputs vertexInput = GetVertexPositionInputs(v.vertex.xyz);
				o.position = vertexInput.positionCS;
                
                o.worldPos = vertexInput.positionWS.xyz;

				VertexNormalInputs normaInput = GetVertexNormalInputs(v.normal);
				o.worldNormal = normaInput.normalWS;
				o.texcoord = v.texcoord;
                return o;  
            }

            half4 frag(v2f i) : COLOR 
            {  
                float2 uv_x = TRANSFORM_TEX(i.worldPos.zy, _Basemap);
                float2 uv_y = TRANSFORM_TEX(i.worldPos.xz, _Basemap);
                float2 uv_z = TRANSFORM_TEX(i.worldPos.xy, _Basemap);

                half4 col_x = tex2D(_Basemap, uv_x);
                half4 col_y = tex2D(_Basemap, uv_y);
				half4 col_z = tex2D(_Basemap, uv_z);

                half3 weights = i.worldNormal;
                weights = abs(weights);
                weights = pow(weights, 64);
                weights = weights / (weights.x + weights.y + weights.z);

                col_x *= weights.x;
                col_y *= weights.y;
                col_z *= weights.z;
            
                Light mainLight = GetMainLight();
                half3 lightDir = mainLight.direction;
                half3 NdotL = max(0, dot(i.worldNormal, lightDir));
                half4 light = half4(NdotL * mainLight.color.xyz + SampleSH(i.worldNormal).xyz, 1);
				 
                // float2 uv = TRANSFORM_TEX(i.worldPos, _Basemap);
                // half4 diffuse = tex2D(_Basemap, uv);
                
                half4 diffuse = col_x + col_y + col_z;

				half4 color = diffuse * light;
				
                return color;
            }
            ENDHLSL  
        }  
    }   
    FallBack Off 
}