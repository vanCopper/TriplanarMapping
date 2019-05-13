Shader "SolarLand/Terrain/Mesh/Tessellation_LWRP" 
{
    Properties 
    {  
        [NoScaleOffset]_Control("Control Map (RGBA)", 2D ) = "red" {}
        _ContrastNear("Blend Contrast Near", Range(0.01, 0.99)) = 0.35
        _ContrastFar("Blend Contrast Far", Range(0.01, 0.99)) = 0.85
        _Splat0("Splat0 (RGB)", 2D) = "white" {}
        [NoScaleOffset]_Splat0NM("Splat0 Normal(RGB)", 2D) = "bump" {}
        [NoScaleOffset]_Splat1("Splat1 (RGB)", 2D) = "white" {}
        [NoScaleOffset]_Splat1NM("Splat1 Normal(RGB)", 2D) = "bump" {}
        [NoScaleOffset]_Splat2("Splat2 (RGB)", 2D) = "white" {}
        [NoScaleOffset]_Splat2NM("Splat2 Normal(RGB)", 2D) = "bump" {}
        [NoScaleOffset]_Splat3("Splat3 (RGB)", 2D) = "white" {}
        [NoScaleOffset]_Splat3NM("Splat3 Normal(RGB)", 2D) = "bump" {}
        _Basemap("Base map (RGB)", 2D ) = "white" {}
        _Distance("Blend Distance", float) = 500
        _Width("Blend Width", float) = 200

        [NoScaleOffset]_DispTex ("Disp Texture", 2D) = "gray" {}
        [NoScaleOffset]_WNTex ("World Space Normal", 2D) = "green" {}
		 
        _Displacement ("Displacement", Range(0, 800)) = 0.3
        _Tessellation ("Tessellation", Range(1, 16) ) = 2
        _TessDistance("Tessellation Distance", Range(10, 1000)) = 1.0
		//_TessInside("Tessellation Inside", Range(0, 10)) = 0.5
    }  
    SubShader {  
        Tags{"RenderType" = "Opaque" "RenderPipeline" = "LightweightPipeline"} 
           
        Pass {  
            Tags { "LightMode" = "LightweightForward" }  
               
            HLSLPROGRAM  
			// feature
			// multi_compile
			
			//unity defined keywords
            #pragma hull hull
            #pragma domain domain
            #pragma multi_compile_fog
            //#pragma fragmentoption ARB_precision_hint_fastest
            //#pragma multi_compile_fwdbase nolightmap nodirlightmap nodynlightmap noforwardadd
           //#pragma exclude_renderers xbox360 xboxone ps3 ps4 psp2 
            #pragma target 4.6

			//#ifdef UNITY_CAN_COMPILE_TESSELLATION
			#pragma vertex tessvert
			//#pragma vertex vert
            #pragma fragment frag
            //#include "UnityCG.cginc"
            //#include "Tessellation.cginc" 
            //#include "Lighting.cginc"  
            //#include "AutoLight.cginc"
            #include "Packages/com.unity.render-pipelines.lightweight/ShaderLibrary/Core.hlsl"
			#include "Packages/com.unity.render-pipelines.lightweight/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.lightweight/ShaderLibrary/Shadows.hlsl"
			sampler2D _Control;
            sampler2D _Splat0;
            sampler2D _Splat1;
            sampler2D _Splat2;
            sampler2D _Splat3;
            sampler2D _Splat0NM;
            sampler2D _Splat1NM;
            sampler2D _Splat2NM;
            sampler2D _Splat3NM;
            half4 _Splat0_ST;
			half4 _Splat1_ST;
            //half4 _Control_TexelSize;
            half _ContrastNear;
            half _ContrastFar;

            sampler2D _Basemap;
            sampler2D _WNTex;
            half4 _Basemap_ST;
            half _Distance;
            half _Width;

            sampler2D _DispTex;  
            half _Tessellation;
            half _TessDistance;
            half _Displacement;
			half _TessInside;
             
            struct a2v 
            {  
                half4 vertex : POSITION;  
                half3 normal : NORMAL;  
                half4 tangent : TANGENT;  
                half4 texcoord : TEXCOORD0;  
                
            };  
               
            struct v2f {  
                half4 vertex        : POSITION;  
                float4 texcoord0    : TEXCOORD0; // xy == uv0, z == depth;; w == blendContrast
                float3 texcoord2    : TEXCOORD2; // xy:uv1, z:fogCoord
				float4 shadowCoord	: TEXCOORD7;
                //UNITY_FOG_COORDS(6) 
                //SHADOW_COORDS(7)
                half2 yUV           : TEXCOORD8;
                half2 xUV           : TEXCOORD9;
                half2 zUV           : TEXCOORD10;
            };  

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //Functions --TODO #include "SolarLandTerrainCommon.cginc"
            half4 ComputeWeights(half4 iWeights, half h0, half h1, half h2, half h3, half contrast)
            {
                // compute weight with height map
                half4 weights = half4(iWeights.x * max(h0,0.001), iWeights.y * max(h1,0.001), iWeights.z * max(h2,0.001), iWeights.w * max(h3,0.001));

                // Contrast weights
                half maxWeight = max(max(weights.x, max(weights.y, weights.z)), weights.w);
                half transition = max(contrast * maxWeight, 0.0001);
                half threshold = maxWeight - transition;
                half scale = 1.0 / transition;
                weights = saturate((weights - threshold) * scale);
                // Normalize weights.
                half weightScale = 1.0f / (weights.x + weights.y + weights.z + weights.w);
                weights *= weightScale;
                return weights;
            }
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
               
            v2f vert(a2v v) 
            {  
                v2f o;
                //o.vertex = UnityObjectToClipPos(v.vertex);
                v.vertex.y += tex2Dlod(_DispTex, v.texcoord).r * _Displacement;
				VertexPositionInputs vertexInput = GetVertexPositionInputs(v.vertex.xyz);
				o.vertex = vertexInput.positionCS;
                o.texcoord0.xy = v.texcoord.xy;
                o.texcoord2.xy = v.texcoord.xy * _Basemap_ST.xy + _Basemap_ST.zw;

                //World position as triplanar UVs
                //half3 worldPos = mul(unity_ObjectToWorld, v.vertex);
				half3 worldPos = vertexInput.positionWS.xyz;
                o.yUV = worldPos.xz;// * _Splat0_ST.xy + _Splat1_ST.zw;
                o.xUV = worldPos.zy;// * _Splat1_ST.xy + _Splat1_ST.zw;
                o.zUV = worldPos.xy;// * _Splat1_ST.xy + _Splat1_ST.zw;

                //Depth blending factor
                //half z = length(UnityObjectToViewPos(v.vertex).xyz);
				half z = length(vertexInput.positionVS.xyz);
                o.texcoord0.z =  1 - saturate((_Distance - z)/_Width);
                o.texcoord0.w = saturate((50 - z)/50) * (_ContrastNear - _ContrastFar) + _ContrastFar;

                //UNITY_TRANSFER_FOG(o,o.vertex);
				o.texcoord2.z = ComputeFogFactor(vertexInput.positionCS.z);
				o.shadowCoord = GetShadowCoord(vertexInput);

                
                //TRANSFER_SHADOW(o)
         
                return o;  
            }

            //#ifdef UNITY_CAN_COMPILE_TESSELLATION
				// Distance based tessellation:
				// Tessellation level is "tess" before "minDist" from camera, and linearly decreases to 1
				// up to "maxDist" from camera.
				float UnityCalcDistanceTessFactor (float4 vertex, float minDist, float maxDist, float tess)
				{
					float3 wpos = mul(unity_ObjectToWorld,vertex).xyz;
					float dist = distance (wpos, _WorldSpaceCameraPos);
					float f = clamp(1.0 - (dist - minDist) / (maxDist - minDist), 0.01, 1.0) * tess;
					return f;
				}

				float4 UnityCalcTriEdgeTessFactors (float3 triVertexFactors)
				{
					float4 tess;
					tess.x = 0.5 * (triVertexFactors.y + triVertexFactors.z);
					tess.y = 0.5 * (triVertexFactors.x + triVertexFactors.z);
					tess.z = 0.5 * (triVertexFactors.x + triVertexFactors.y);
					tess.w = (triVertexFactors.x + triVertexFactors.y + triVertexFactors.z) / 3.0f;
					return tess;
				}

				float4 UnityDistanceBasedTess (float4 v0, float4 v1, float4 v2, float minDist, float maxDist, float tess)
				{
					float3 f;
					f.x = UnityCalcDistanceTessFactor (v0,minDist,maxDist,tess);
					f.y = UnityCalcDistanceTessFactor (v1,minDist,maxDist,tess);
					f.z = UnityCalcDistanceTessFactor (v2,minDist,maxDist,tess);

					return UnityCalcTriEdgeTessFactors (f);
				}

				
                struct TessVertex 
                {
                    half4 vertex    : INTERNALTESSPOS;
                    half4 texcoord  : TEXCOORD0; 
                    half3 normal : Normal; 
                };
                struct OutputPatchConstant 
                {
                    half edge[3]         : SV_TessFactor; //Èý½ÇÃæµÄÏ¸·ÖÒò×Ó
                    half inside          : SV_InsideTessFactor; //Èý½ÇÐÎÄÚ²¿µÄÏ¸·ÖÒò×Ó
                };

                TessVertex tessvert (a2v v) 
                {
                    TessVertex o;
                    o.vertex = v.vertex;
                    o.texcoord = v.texcoord;
                    o.normal = v.normal;
//                    o.vertex.y += tex2Dlod(_DispTex, v.texcoord).r * _Displacement;
                    return o;
                }

                half Tessellation(TessVertex v)
                {
                    return _Tessellation;
                }

                half4 Tessellation(TessVertex v, TessVertex v1, TessVertex v2)
                {
					//float4 UnityDistanceBasedTess (float4 v0, float4 v1, float4 v2, float minDist, float maxDist, float tess)
					//real3 GetDistanceBasedTessFactor(real3 p0, real3 p1, real3 p2, real3 cameraPosWS, real tessMinDist, real tessMaxDist)
					//float3 cameraPosWS = GetCameraPositionWS();
                    return UnityDistanceBasedTess(v.vertex, v1.vertex, v2.vertex, 10, _TessDistance, _Tessellation);
                }
                OutputPatchConstant hullconst (InputPatch<TessVertex,3> v) 
                {
                    OutputPatchConstant o = (OutputPatchConstant)0;
                    half4 ts = Tessellation( v[0], v[1], v[2] );
                    o.edge[0] = ts.x;
                    o.edge[1] = ts.y; 
                    o.edge[2] = ts.z;
                    o.inside = ts.w;
					return o; 
                }
                [domain("tri")]
                [partitioning("fractional_odd")]
                [outputtopology("triangle_cw")]
                [patchconstantfunc("hullconst")]
                [outputcontrolpoints(3)]
                TessVertex hull (InputPatch<TessVertex,3> v, uint id : SV_OutputControlPointID) 
                {
                     return v[id];
                }

                [domain("tri")]
                v2f domain (OutputPatchConstant tessFactors, const OutputPatch<TessVertex,3> vi, half3 bary : SV_DomainLocation) {
                    a2v v = (a2v)0;
                    v.vertex = vi[0].vertex*bary.x + vi[1].vertex*bary.y + vi[2].vertex*bary.z;
                    v.texcoord = vi[0].texcoord*bary.x + vi[1].texcoord*bary.y + vi[2].texcoord*bary.z;
                    v2f o = vert(v);
                    return o;
                }
            //#endif

            #define TRIPLANAR_BLEND_SHARPNESS 180
            half4 frag(v2f i) : COLOR 
            {  
                //fixed shadow = SHADOW_ATTENUATION(i);
				half shadow = SampleScreenSpaceShadowmap(i.shadowCoord);
                half3 wNormal = (tex2D(_WNTex, i.texcoord0.xy).xyz * 2 - 1);
               // half3 wNormal = tex2D(_WNTex, i.yUV).xyz * 2 - 1;
                half4 control = tex2D(_Control, i.texcoord0.xy);// + i.texcoord0.xy * _Control_TexelSize.xy);
                half3 basemap = tex2D(_Basemap, i.texcoord2.xy).xyz;
				//return half4(wNormal,1);
                half4 splat0 = tex2D(_Splat0, i.yUV);

				//////////////////////////////////////////////////////////////////////////////////////
                //Triplanar mapping Diffuse.
                half3 blendWeights = pow(abs(wNormal), 2);
                //half3 blendWeights = wNormal*0.25 + (1-0.25)*0.5;
                 blendWeights = blendWeights / (blendWeights.x + blendWeights.y + blendWeights.z);
				//blendWeights = abs(wNormal);
				blendWeights = saturate(blendWeights);
                half4 yDiff = tex2D(_Splat1, i.yUV);//*0.0234375
                half4 xDiff = tex2D(_Splat1, i.xUV);
                half4 zDiff = tex2D(_Splat1, i.zUV);
				half4 splat1 = xDiff * blendWeights.x + yDiff * blendWeights.y + zDiff * blendWeights.z;
                //End of Triplanar mapping Diffuse 

				//float3 blendWeights = abs(wNormal);
				//blendWeights = normalize(max(blendWeights, 0.00001));
				//blendWeights = pow(max(blendWeights, 0), 8);
				//float b = (blendWeights.x + blendWeights.y + blendWeights.z);
				//blendWeights = blendWeights / float3(b,b,b);
				//blendWeights = saturate(blendWeights);
				//half4 yDiff = tex2D(_Splat1, i.yUV*0.05);
                //half4 xDiff = tex2D(_Splat1, i.xUV*0.05);
                //half4 zDiff = tex2D(_Splat1, i.zUV*0.05);
				//half4 splat1 = float4(xDiff.xyz * blendWeights.x + yDiff.xyz * blendWeights.y + zDiff.xyz * blendWeights.z, 1);
				//////////////////////////////////////////////////////////////////////////////////////////////

                half4 splat2 = tex2D(_Splat2, i.yUV);
                half4 splat3 = tex2D(_Splat3, i.yUV);

                //Computer Weights via Heightmaps(splatN.a).
                half4 heightWeights = ComputeWeights(control, splat0.a, splat1.a, splat2.a, splat3.a, i.texcoord0.w);

                half3 mixedNormal;
                mixedNormal =  UnpackNormal(tex2D(_Splat0NM, i.yUV)) * heightWeights.r;
                mixedNormal += UnpackNormal(tex2D(_Splat2NM, i.yUV)) * heightWeights.b;
                mixedNormal += UnpackNormal(tex2D(_Splat3NM, i.yUV)) * heightWeights.a;

                //Triplanar mapping NormalMap. Tangent space normal maps
                half3 tNormalY = mixedNormal + UnpackNormal(tex2D(_Splat1NM, i.yUV)) * heightWeights.g;
                half3 tNormalX = mixedNormal + UnpackNormal(tex2D(_Splat1NM, i.xUV)) * heightWeights.g;
                half3 tNormalZ = mixedNormal + UnpackNormal(tex2D(_Splat1NM, i.zUV)) * heightWeights.g;

                // Swizzle tangent normals into world space and zero out "z"
                half3 normalX = half3(0.0, tNormalX.yx);
                half3 normalY = half3(tNormalY.x, 0.0, tNormalY.y);
                half3 normalZ = half3(tNormalZ.xy, 0.0);

                // Triblend normals and add to world normal
                half3 triplanarNormal = normalX * blendWeights.x + normalY * blendWeights.y + normalZ * blendWeights.z + wNormal;
                half3 worldNormal = triplanarNormal;
                //End of Triplanar mapping NormalMap.

                //half3 lightDir = _WorldSpaceLightPos0.xyz;
				Light mainLight = GetMainLight();
                half3 lightDir = mainLight.direction;
                half3 NdotL = max(0, dot(worldNormal, lightDir));
                half3 light = NdotL * mainLight.color.xyz + SampleSH(worldNormal).xyz;
                
				 
                half3 diffuse = splat0.xyz * heightWeights.r + splat1.xyz * heightWeights.g + splat2.xyz * heightWeights.b + splat3.xyz * heightWeights.a;
                diffuse = lerp(diffuse, basemap, i.texcoord0.z);
				// ÒõÓ°Ã²ËÆ»¹ÓÐµãÎÊÌâ
                //half4 color = half4(diffuse * light * shadow, 1);
				half4 color = half4(diffuse * light, 1);
				//half4 color = half4(diffuse * light, 1);
                //UNITY_APPLY_FOG(i.fogCoord, color);
                color.xyz = MixFog(color.xyz, i.texcoord2.z); 
                return color;//half4(blendWeights.y,0, 0, 1);  
            }
            ENDHLSL  
        }  
    }   
    FallBack Off 
}